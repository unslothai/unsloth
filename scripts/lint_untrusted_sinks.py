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
        # Weight files. The tensor NAMES in a downloaded checkpoint are attacker-chosen
        # just as a config value is, and they get used as attribute names and as paths.
        # Without these the only chain the scanner could see into an adapter loader was
        # the config beside it, so a finding on a tensor name read as if it came from
        # `adapter_config.json`.
        "mlx.core.load",
        "safetensors.torch.load_file",
        "safetensors.numpy.load_file",
        "safe_open",
        "safetensors.safe_open",
        "load_file",
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
UNTRUSTED_METHODS = frozenset(
    # read_bytes beside read_text: `pickle.loads(Path(download).read_bytes())` is the
    # shorter spelling of the handle shape and lost the taint the path already carried.
    {"read", "read_text", "read_bytes", "readlines", "readline", "json"}
)

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
    # `path` is the keyword this takes, and an empty set meant the named spelling was
    # inspected neither positionally nor by keyword.
    "pydoc.locate": ((0,), frozenset({"path"})),
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
    # No bare `path.append` entry: `from sys import path` canonicalises to
    # `sys.path.append` through the import table, so the bare spelling only ever matched
    # an unrelated local list called `path`, and reporting `path = []; path.append(parsed)`
    # blocked ordinary code.
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
    # The async spellings run a program exactly as the blocking ones do.
    "asyncio.create_subprocess_exec": ((0,), frozenset({"program", "executable"})),
    "asyncio.create_subprocess_shell": ((0,), frozenset({"cmd"})),
    "create_subprocess_exec": ((0,), frozenset({"program", "executable"})),
    "create_subprocess_shell": ((0,), frozenset({"cmd"})),
    # executable beside args: it names the program that actually runs, so a fixed argv
    # with a tainted executable executes the tainted one. Only args was inspected.
    "subprocess.run": ((0,), frozenset({"args", "executable"})),
    "subprocess.call": ((0,), frozenset({"args", "executable"})),
    "subprocess.check_call": ((0,), frozenset({"args", "executable"})),
    "subprocess.check_output": ((0,), frozenset({"args", "executable"})),
    "subprocess.Popen": ((0,), frozenset({"args", "executable"})),
    # The exec family replaces this process with the named program, so a tainted path
    # here IS execution with no shell in between. `os.posix_spawn` takes the same
    # argument shape; the spawn family puts a mode first, so the path sits at 1.
    "os.execv": ((0,), frozenset()),
    "os.execve": ((0,), frozenset()),
    "os.execvp": ((0,), frozenset()),
    "os.execvpe": ((0,), frozenset()),
    "os.execl": ((0,), frozenset()),
    "os.execle": ((0,), frozenset()),
    "os.execlp": ((0,), frozenset()),
    "os.execlpe": ((0,), frozenset()),
    "os.posix_spawn": ((0,), frozenset()),
    "os.posix_spawnp": ((0,), frozenset()),
    "os.spawnv": ((1,), frozenset()),
    "os.spawnve": ((1,), frozenset()),
    "os.spawnvp": ((1,), frozenset()),
    "os.spawnvpe": ((1,), frozenset()),
    "os.spawnl": ((1,), frozenset()),
    "os.spawnle": ((1,), frozenset()),
    "os.spawnlp": ((1,), frozenset()),
    "os.spawnlpe": ((1,), frozenset()),
    # Deserialisers that construct arbitrary objects.
    # `file` is the keyword both loaders take, and an empty set meant the named
    # spelling was inspected neither positionally nor by keyword.
    "pickle.load": ((0,), frozenset({"file"})),
    "pickle.loads": ((0,), frozenset({"data"})),
    "dill.load": ((0,), frozenset({"file"})),
    "dill.loads": ((0,), frozenset({"str"})),
    # Not `torch.load`: that one depends on `weights_only`, which is a value and not a
    # position, so it has its own check below.
}

# `torch.load(downloaded, weights_only = False)` runs the pickle-based loader and can
# construct arbitrary objects out of the checkpoint, which is the same execution the
# `pickle` entries above exist for. Kept out of the table because the decision is the
# value of `weights_only` rather than an argument position.
TORCH_LOAD_NAMES = frozenset({"torch.load", "load"})
TORCH_LOAD_SINK = "torch.load(weights_only = False)"
# How an alias of `torch.load` is recorded in the sink-alias tables. Like "getattr" it is
# not a SINKS key, so the table check skips it and the conditional check reads it.
TORCH_LOAD_ALIAS = "torch.load"
_NOT_TABLE_SINKS = frozenset({"getattr", TORCH_LOAD_ALIAS})

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


# node -> its dotted text. Keyed by the node OBJECT, not by id(), so a tree freed
# between scans cannot have its id reused by a later one. Cleared at the start of every
# scan. This is the hottest lookup in the whole run by a wide margin: the fixpoint
# re-walks each body until its state settles, so the same nodes are read millions of
# times and the answer never changes.
_CALL_NAMES: dict = {}


def _call_name(node: ast.AST) -> str:
    """The dotted text of a call target, or "" when it is not a plain dotted name."""
    cached = _CALL_NAMES.get(node)
    if cached is not None:
        return cached
    name = _call_name_uncached(node)
    _CALL_NAMES[node] = name
    return name


def _call_name_uncached(node: ast.AST) -> str:
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


# (name, the table's identity) -> the entry it matched. Only the long-lived tables
# registered below are cached: several call sites pass a set literal built on the spot,
# and once such a set is freed another can land on the same address, so caching by id
# would hand back the previous table's answer. That is exactly what happened on the
# first attempt, and the self-test caught it by missing a download-derived sys.path
# entry while the tier A count fell from 77 to 60.
_MATCHES: dict = {}
_CACHEABLE_TABLES: set = set()


def _matches(name: str, table) -> str | None:
    """Match a dotted callee against a table on its trailing segments.

    `json.load`, `js.load` and a bare `load` all have to hit the same entry, because
    which spelling appears is an import style and not a security property.
    """
    if not name:
        return None
    if id(table) not in _CACHEABLE_TABLES:
        return _matches_uncached(name, table)
    key = (name, id(table))
    if key in _MATCHES:
        return _MATCHES[key]
    answer = _matches_uncached(name, table)
    _MATCHES[key] = answer
    return answer


def _matches_uncached(name: str, table) -> str | None:
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


def _norm_body_hash(statements: list) -> str:
    """`_norm_hash` over a list of statements, for module and class bodies."""
    return _norm_hash(ast.Module(body = list(statements), type_ignores = []))


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
        # Filled by `scan` once every file is parsed, so resolution can follow a package
        # re-export one hop to the file that actually defines the helper.
        self.facts: dict = {}
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

    def follow_reexport(self, file: Path, qualname: str) -> tuple | None:
        """`from pkg import parse`, where `pkg/__init__.py` does `from .parser import parse`.

        Resolution stopped at the package `__init__.py` and handed back a qualname that
        file does not define, so `pkg.parser.parse` was never analysed and taint stopped
        at the package API, which is how most of this tree imports its own helpers.

        Chased rather than one hop: `pkg/__init__` re-exporting from `pkg.api/__init__`
        which re-exports from `pkg.api.impl` is two layers, and stopping after the first
        handed back a file that does not define the symbol either, so the hop was
        declined and the implementation was never analysed. Bounded with a seen set, so
        a package that re-exports in a circle terminates.
        """
        seen: set = set()
        current = (file, qualname)
        for _ in range(_REEXPORT_HOPS):
            if current in seen:
                return None
            seen.add(current)
            facts = self.facts.get(current[0])
            if facts is None or not current[1] or current[1] in facts.functions:
                return None if current == (file, qualname) else current
            step = None
            for dotted in facts._targets(current[1]):
                target, module = self.resolve_module(dotted)
                if target is None or target == current[0]:
                    continue
                inner = (
                    dotted[len(module) + 1 :] if module and dotted.startswith(module + ".") else ""
                )
                if not inner:
                    continue
                defining = self.facts.get(target)
                if defining is None or inner in defining.functions:
                    return (target, inner)
                step = (target, inner)
                break
            if step is None:
                return None
            current = step
        return None

    def resolve(self, dotted: str) -> Path | None:
        """The file defining `dotted`, walking up so `a.b.func` finds module `a.b`."""
        return self.resolve_module(dotted)[0]

    def resolve_module(self, dotted: str) -> tuple:
        """(file, the module spelling that matched), walking up from the longest prefix.

        The spelling matters and `files[file]` cannot supply it. A file under
        studio/backend is indexed under both roots, and `files` keeps the shorter name, so
        stripping `utils.parser` off `studio.backend.utils.parser.parse` left an empty
        qualname: the callee did not resolve to any function and taint returned by a
        backend helper never reached its caller in the CLI. Returning what actually matched
        keeps the two spellings interchangeable.
        """
        segments = dotted.split(".")
        while segments:
            candidate = ".".join(segments)
            hit = self.modules.get(candidate)
            if hit is not None:
                return hit, candidate
            segments.pop()
        return None, ""


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
        # qualname -> name of its *args parameter, so overflow positionals land
        self.varargs: dict[str, str] = {}
        # Class names defined in this file, so `parser = Parser()` can be recognised as
        # constructing one and `parser.parse(...)` resolved to `Parser.parse`.
        self.classes: set = set()
        # class name -> the base names it declares, so `super().m()` can be resolved
        self.bases: dict = {}
        # qualname -> names it declares global, so a write lands in module state
        self.globals_declared: dict = {}
        # Module-scope `loader = importlib.import_module` and `parser = Parser()`. These
        # were recorded on the visitor that scanned the module body and discarded before
        # any function was scanned, so a function calling the alias read as clean while
        # the identical local alias was caught.
        self.module_sink_aliases: dict = {}
        # name -> every class it was seen constructed from. A tuple, not one name:
        # `runner = Safe()` then `runner = Dirty()` is valid sequential code and
        # keeping only the first resolved calls to the wrong class, so an execution
        # sink in the second one received tainted data with nothing reported.
        self.module_instances: dict = {}
        # Module-scope `runner = execute`, for the same reason. A tuple of every
        # callable the name was bound to, since the analysis is flow-insensitive and
        # picking one of them is a guess.
        self.module_callable_aliases: dict = {}
        # Module-scope `decode = json.loads`, the source counterpart of the sink table.
        self.module_source_aliases: dict = {}
        # Module-scope `ENABLED = True`, forwarded as `trust_remote_code = ENABLED`.
        self.module_true_names: set = set()
        # Module-scope `add_path = functools.partial(sys.path.insert, 0)`: how many
        # positional arguments the wrapper no longer takes, seeded into every function.
        self.module_alias_offsets: dict = {}
        # Pure lookups over tables that are fixed once collection finishes. Memoised
        # because the fixpoint reads them millions of times per run and the answers
        # cannot change: the gate has to be fast enough to sit in CI.
        self._target_cache: dict = {}
        self._canonical_cache: dict = {}
        self._construction_cache: dict = {}
        self._methods_cache: dict = {}
        self._ancestor_cache: dict = {}
        # Aliases bound by a plain `import x` / `import x as y`, which are modules by
        # construction. `getattr(namespace, parsed)` on one is unsafe reflection, and the
        # name heuristic alone missed every module this tree imports under a name it does
        # not happen to list.
        self.module_aliases: set = set()
        # Module-level dicts whose values are all literals. A lookup in one is a
        # validated translation, not a passthrough: the result can only be one of the
        # constants written in the source, whatever key the attacker supplies.
        self.constant_maps: set = set()
        # Modules pulled in with `from x import *`, expanded against the exporting
        # file once every file has been parsed.
        self.wildcard_bases: list = []
        self._wildcard_cache: dict = {}
        # `__all__`, when the file declares one. None means "not declared", which is a
        # different answer from "declared empty".
        self.exported: set | None = None
        # Qualnames whose immediate enclosing scope is a class and which are not
        # `@staticmethod`. Python does not require the receiver to be named `self`, so
        # reading method status off the first parameter misclassified `def execute(this,
        # command)` and bound a tainted argument to the receiver instead of the parameter.
        self.methods: set = set()
        # Qualnames decorated with `property`, so `cfg.module` can be read as the call it
        # really is rather than as a plain attribute.
        self.properties: set = set()
        # Qualnames that are lambdas, whose body IS their return expression.
        self.lambdas: set = set()
        self._collect()
        self._collect_module_bindings()

    def _collect(self) -> None:
        scope: list[str] = []
        # Whether each open scope is a class, so the function below knows if its
        # immediate parent is one. A nested helper inside a method is not a method.
        kinds: list[str] = []

        def walk(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        if alias.asname:
                            self._bind_import(alias.asname, alias.name)
                            self.module_aliases.add(alias.asname)
                        else:
                            # `import pkg.parser` binds `pkg`, not `pkg.parser`. Recording
                            # the full dotted name against the head made resolution append
                            # the original tail to it a second time, so `pkg.parser.parse`
                            # resolved as `pkg.parser.parser.parse` and did not resolve at
                            # all: taint returned by that helper never reached its caller.
                            head = alias.name.split(".")[0]
                            self._bind_import(head, head)
                            self.module_aliases.add(head)
                elif isinstance(child, ast.ImportFrom):
                    base = self._absolute(child)
                    for alias in child.names:
                        if alias.name == "*":
                            # `from producer import *` binds every exported name, and
                            # recording the literal `*` meant a later call to one of them
                            # resolved to nothing: taint never reached the helper. The
                            # names come from the exporting file, which is not parsed
                            # yet, so the base is kept and expanded on demand.
                            if base:
                                self.wildcard_bases.append(base)
                            continue
                        target = f"{base}.{alias.name}" if base else alias.name
                        self._bind_import(alias.asname or alias.name, target)
                elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                    assigned = [child.target] if isinstance(child, ast.AnnAssign) else child.targets
                    if isinstance(child.value, ast.Lambda):
                        for target in assigned:
                            if isinstance(target, ast.Name):
                                self._index_lambda(target.id, child.value, ".".join(scope))
                    walk(child)
                    continue
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    in_class = bool(kinds) and kinds[-1] == "class"
                    scope.append(child.name)
                    kinds.append("function")
                    qualname = ".".join(scope)
                    if in_class and not any(
                        _call_name(decorator).rpartition(".")[2] == "staticmethod"
                        for decorator in child.decorator_list
                    ):
                        self.methods.add(qualname)
                    if in_class and any(
                        _call_name(decorator).rpartition(".")[2] in ("property", "cached_property")
                        for decorator in child.decorator_list
                    ):
                        self.properties.add(qualname)
                    self.functions[qualname] = child
                    self.params[qualname] = _param_names(child)
                    self.contexts[qualname] = _norm_hash(child)
                    if child.args.kwarg is not None:
                        self.star_kwargs[qualname] = child.args.kwarg.arg
                    if child.args.vararg is not None:
                        self.varargs[qualname] = child.args.vararg.arg
                    # This function's own body only. `ast.walk` descended into nested
                    # functions, so a `global command` declared by an inner helper was
                    # recorded for the enclosing one, and the outer function assigning
                    # its own local `command` then poisoned the module global: an
                    # unrelated function executing the safe global failed the gate.
                    declared = _declared_here(child, ast.Global)
                    if declared:
                        self.globals_declared[qualname] = declared
                    walk(child)
                    scope.pop()
                    kinds.pop()
                    continue
                elif isinstance(child, ast.ClassDef):
                    self.classes.add(child.name)
                    self.classes.add(".".join(scope + [child.name]))
                    declared_bases = []
                    for base in child.bases:
                        base_name = _call_name(base)
                        if base_name:
                            # Both spellings: the final segment is what a class declared
                            # in this file is indexed under, and the qualified form is
                            # what the import table can resolve. Stripping every base
                            # left `class Child(producer.Base)` with only `Base`, which
                            # `_targets` has no binding for, so inherited sinks behind
                            # the `import producer` style resolved to nothing.
                            declared_bases.append(base_name.rpartition(".")[2])
                            if base_name not in declared_bases:
                                declared_bases.append(base_name)
                    if declared_bases:
                        self.bases[".".join(scope + [child.name])] = declared_bases
                        self.bases.setdefault(child.name, declared_bases)
                    scope.append(child.name)
                    kinds.append("class")
                    walk(child)
                    scope.pop()
                    kinds.pop()
                    continue
                walk(child)

        walk(self.tree)

    def _index_lambda(
        self,
        name: str,
        node: ast.Lambda,
        scope: str = "",
    ) -> None:
        """`runner = lambda command: subprocess.run(command)` is a callable with a body.

        `_call_name` cannot reduce a Lambda to a name, so the assignment was discarded
        and a call through the name propagated nothing: the sink inside saw an unbound,
        clean parameter. Indexed like any other function so the ordinary machinery
        resolves it, including the receiver-free argument mapping.
        """
        qualname = f"{scope}.{name}" if scope else name
        if qualname in self.functions:
            return
        self.functions[qualname] = node
        self.params[qualname] = _param_names(node)
        self.contexts[qualname] = _norm_hash(node)
        if node.args.kwarg is not None:
            self.star_kwargs[qualname] = node.args.kwarg.arg
        if node.args.vararg is not None:
            self.varargs[qualname] = node.args.vararg.arg
        self.lambdas.add(qualname)

    def _note_module_partial(self, names: list, value: ast.Call) -> bool:
        """`functools.partial(...)` at module scope, recorded like the local form.

        Returns whether the call was a partial, so the caller can stop. The pre-bound
        arguments themselves are checked where the assignment is walked, which is the
        module body; what has to be collected here is the identity and the layout, which
        is what every function visitor is seeded from.
        """
        if (
            _matches_any(self.canonicals(_call_name(value.func)), {"functools.partial", "partial"})
            is None
        ):
            return False
        if not value.args:
            return True
        inner = _call_name(value.args[0])
        prebound = len(value.args) - 1 + self.module_alias_offsets.get(inner, 0)
        direct = _matches_any(self.canonicals(inner), SINKS)
        held = [direct] if direct is not None else list(self.module_sink_aliases.get(inner) or ())
        if not held:
            relayed = self.module_callable_aliases.get(inner) or ()
            alias = self.callable_alias(inner)
            candidates = list(relayed) or ([alias] if alias else [])
            for name in names:
                for candidate in candidates:
                    self.module_callable_aliases[name] = _with(
                        self.module_callable_aliases.get(name), candidate
                    )
                if candidates and prebound:
                    self.module_alias_offsets[name] = prebound
            return True
        for name in names:
            for candidate in held:
                self.module_sink_aliases[name] = _with(
                    self.module_sink_aliases.get(name), candidate
                )
            if prebound:
                self.module_alias_offsets[name] = prebound
        return True

    def _note_constant_map(self, names: list, value: ast.AST) -> None:
        """`_DTYPES = {"F32": "float32", ...}`: a fixed translation table.

        `getattr(mx, _DTYPES.get(meta["dtype"], "float32"))` reads as a dynamic name but
        can only produce one of the literals above it, which is what validating at the
        producer looks like. Reporting those is how a gate earns its way into being
        switched off, so the lookup is treated as the sanitiser it is.
        """
        if not isinstance(value, ast.Dict) or not value.keys:
            return
        if any(key is None for key in value.keys):
            return
        if not all(isinstance(item, ast.Constant) for item in value.values):
            return
        for name in names:
            self.constant_maps.add(name)

    def _collect_module_bindings(self) -> None:
        """Module-scope aliases of sinks and constructions, after imports are known."""
        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.AnnAssign):
                targets, value = ([node.target], node.value)
            elif isinstance(node, ast.Assign):
                targets, value = (node.targets, node.value)
            else:
                continue
            if value is None:
                continue
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if not names:
                continue
            if "__all__" in names and isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                self.exported = {
                    item.value
                    for item in value.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                }
            if isinstance(value, ast.Constant) and value.value is True:
                self.module_true_names.update(names)
            self._note_constant_map(names, value)
            if isinstance(value, ast.Lambda):
                for name in names:
                    self._index_lambda(name, value)
                continue
            if isinstance(value, ast.Call):
                # A partial at module scope is the same wrapper as a local one, and this
                # branch only ever asked whether the call constructs a class and then
                # continued, so neither the wrapped sink nor its offset was ever seeded
                # into a function visitor: a module-level `add_path =
                # partial(sys.path.insert, 0)` called from a function reported nothing.
                if self._note_module_partial(names, value):
                    continue
                # Through the shared resolver, so an imported first-party class is
                # accepted here too. Taking only classes declared in this file left
                # `from producer import Parser; parser = Parser()` at module scope
                # without a type, so a function calling `parser.parse(blob)` resolved
                # to nothing and a sink inside that method was reported nowhere.
                constructed = self.resolve_construction(_call_name(value.func))
                if constructed:
                    for name in names:
                        self.module_instances[name] = _with(
                            self.module_instances.get(name), constructed
                        )
                continue
            referenced = _call_name(value)
            if not referenced:
                continue
            # `ENABLED = True` then `REMOTE = ENABLED`, the module-scope counterpart of
            # the local alias chain. Only the literal assignment was collected, so one
            # hop was enough to hide remote-code enablement from the gate.
            if referenced in self.module_true_names:
                self.module_true_names.update(names)
                continue
            # Chains collected at module scope: `loader = import_module` then
            # `invoke = loader`. Only imports were consulted, so function visitors were
            # seeded with the first alias and never the second.
            chained = self.module_sink_aliases.get(referenced)
            if chained:
                for name in names:
                    for candidate in chained:
                        self.module_sink_aliases[name] = _with(
                            self.module_sink_aliases.get(name), candidate
                        )
                continue
            carried = self.module_source_aliases.get(referenced)
            if carried is not None:
                for name in names:
                    self.module_source_aliases.setdefault(name, carried)
                continue
            relayed = self.module_callable_aliases.get(referenced)
            if relayed:
                for name in names:
                    for candidate in relayed:
                        self.module_callable_aliases[name] = _with(
                            self.module_callable_aliases.get(name), candidate
                        )
                continue
            sink = _matches_any(self.canonicals(referenced), SINKS)
            if sink is None and _matches_any(self.canonicals(referenced), {"getattr"}):
                sink = "getattr"
            if sink is None and _matches_any(self.canonicals(referenced), {"torch.load"}):
                sink = TORCH_LOAD_ALIAS
            if sink is not None:
                for name in names:
                    self.module_sink_aliases[name] = _with(self.module_sink_aliases.get(name), sink)
                continue
            source = _matches_any(self.canonicals(referenced), UNTRUSTED_CALLS)
            if source is not None:
                for name in names:
                    self.module_source_aliases.setdefault(name, source)
                continue
            alias = self.callable_alias(referenced)
            if alias:
                for name in names:
                    self.module_callable_aliases[name] = _with(
                        self.module_callable_aliases.get(name), alias
                    )

    def _bind_import(self, alias: str, target: str) -> None:
        self.imports.setdefault(alias, set()).add(target)

    def _targets(self, alias: str) -> list[str]:
        """Every module one alias can mean in this file, in a deterministic order.

        Memoised: the import table is complete before any analysis starts and never
        changes after that, and this is read tens of millions of times per run.
        """
        cached = self._target_cache.get(alias)
        if cached is None:
            cached = sorted(self.imports.get(alias, ()))
            self._target_cache[alias] = cached
        if cached or not self.wildcard_bases:
            return cached
        return self._wildcard_targets(alias)

    def _wildcard_targets(self, alias: str) -> list[str]:
        """`from producer import *` then a call to `execute`.

        Resolved against the exporting file rather than guessed, so a name that file
        does not export stays unresolved. Memoised only once the index has published
        every file's facts, because before that the answer would be a false negative
        cached forever.
        """
        cached = self._wildcard_cache.get(alias)
        if cached is not None:
            return cached
        index = getattr(self, "index", None)
        if index is None or not getattr(index, "facts", None):
            return []
        found: list[str] = []
        for base in self.wildcard_bases:
            file, module = index.resolve_module(base)
            if file is None or module != base:
                continue
            defining = index.facts.get(file)
            if defining is None or alias not in defining.exported_names():
                continue
            found.append(f"{base}.{alias}")
        found = sorted(set(found))
        self._wildcard_cache[alias] = found
        return found

    def exported_names(self) -> set:
        """What `import *` from this file binds: `__all__` if declared, else the publics."""
        if self.exported is not None:
            return self.exported
        # Imported public names too: `from impl import execute` in the exporting file is
        # bound by a consumer's `import *` exactly like a local definition, and leaving
        # it out meant the re-exported helper never resolved.
        return {
            name
            for name in list(self.functions) + list(self.classes) + list(self.imports)
            if "." not in name and not name.startswith("_")
        }

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
        cached = self._canonical_cache.get(name)
        if cached is not None:
            return cached
        head, separator, tail = name.partition(".")
        targets = self._targets(head)
        if not targets:
            answer = [name]
        else:
            answer = [f"{dotted}.{tail}" if separator else dotted for dotted in targets]
        self._canonical_cache[name] = answer
        return answer

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

    def resolve_construction(
        self,
        constructed: str,
        scope: str = "",
    ) -> str | None:
        """What `parser = Parser()` built: a local class name, or a dotted import target.

        Shared by the module-scope collector and the per-function one so the two cannot
        drift: an instance built at module scope has to resolve the same way one built
        inside a function does. The dotted form is what the resolver tells apart from a
        local class name.
        """
        if not constructed:
            return None
        key = (constructed, scope)
        if key in self._construction_cache:
            return self._construction_cache[key]
        answer = self._resolve_construction(constructed, scope)
        self._construction_cache[key] = answer
        return answer

    def _resolve_construction(self, constructed: str, scope: str) -> str | None:
        # Innermost scope first, so a class declared inside a function resolves to the
        # qualified `load.Runner` its methods are indexed under. The bare spelling
        # matched `Runner`, which is also in the table, and the method lookup then
        # searched for `Runner.execute` while the index holds `load.Runner.execute`.
        parts = scope.split(".") if scope else []
        while parts:
            qualified = ".".join(parts + [constructed])
            if qualified in self.classes:
                return qualified
            parts.pop()
        for candidate in (constructed, constructed.rpartition(".")[2]):
            if candidate and candidate in self.classes:
                return candidate
        head, _, tail = constructed.partition(".")
        for dotted in self._targets(head):
            full = f"{dotted}.{tail}" if tail else dotted
            file, _module = self.index.resolve_module(full)
            if file is not None:
                return full
        return None

    def _methods_on(
        self,
        constructed: str,
        tail: str,
        seen: set | None = None,
    ) -> list:
        """`tail` resolved on a class: declared, or inherited at any depth, local or not.

        The whole chain, not just the direct bases. With `Root.execute`, `Mid(Root)` and
        `Child(Mid)`, a call on a `Child` resolved nowhere, so taint never reached the
        sink in `Root.execute`. A `seen` set because a declaration cycle must not spin.
        """
        if seen is None:
            cached = self._methods_cache.get((constructed, tail))
            if cached is not None:
                return cached
            answer = self._methods_on(constructed, tail, set())
            self._methods_cache[(constructed, tail)] = answer
            return answer
        key = (str(self.path), constructed)
        if key in seen:
            return []
        seen.add(key)
        candidate = f"{constructed}.{tail}"
        if candidate in self.functions:
            return [(self.path, candidate)]
        return self._inherited_methods_on(constructed, tail, seen)

    def _inherited_methods_on(
        self,
        constructed: str,
        tail: str,
        seen: set | None = None,
    ) -> list:
        """`tail` resolved on the BASES of a class, skipping its own declaration.

        What `super().m()` needs: resolving from the class itself would find the
        overriding method that contains the `super()` call and never reach the base.
        """
        if seen is None:
            seen = set()
        candidate = f"{constructed}.{tail}"
        # Inherited rather than declared on the constructed class. The `self.m()`
        # spelling already walked the bases, so a call through an instance gave up where
        # a call from inside the class did not.
        inherited = [
            f"{base}.{tail}"
            for base in self.bases.get(constructed, ())
            if f"{base}.{tail}" in self.functions
        ]
        if inherited:
            return [(self.path, qualname) for qualname in inherited]
        # A first-party base imported from another module. The lookup searched this file
        # only, and most base classes in this tree are imported.
        crossed = self._imported_base_method(constructed, tail, seen)
        if crossed:
            return crossed
        # Further up a local chain: a grandparent that declares the method.
        for base in self.bases.get(constructed, ()):
            if base == constructed:
                continue
            deeper = self._methods_on(base, tail, seen)
            if deeper:
                return deeper
        # An imported class: the instance carries the dotted target instead.
        file, module = self.index.resolve_module(candidate)
        if file is not None and module and candidate.startswith(module + "."):
            qualname = candidate[len(module) + 1 :]
            if qualname:
                return [(file, qualname)]
        return []

    def _imported_base_method(self, constructed: str, tail: str, seen: set) -> list:
        """`class Child(Base)` where `Base` came from another first-party module.

        Recurses into the defining file, so a chain that crosses a file boundary more
        than once still resolves.
        """
        for base in self.bases.get(constructed, ()):
            for dotted in self._targets(base):
                full = f"{dotted}.{tail}"
                file, module = self.index.resolve_module(full)
                if file is None or not module or not full.startswith(module + "."):
                    continue
                qualname = full[len(module) + 1 :]
                defining = self.index.facts.get(file)
                if qualname and (defining is None or qualname in defining.functions):
                    return [(file, qualname)]
                if defining is None or defining is self:
                    continue
                # The base is there but does not declare the method either, so keep
                # walking from the base's own file.
                owner = dotted.rpartition(".")[2]
                deeper = defining._methods_on(owner, tail, seen)
                if deeper:
                    return deeper
        return []

    def _base_chain(
        self,
        class_name: str,
        seen: set | None = None,
    ) -> list:
        """Declared bases of `class_name`, transitively, as this file records them."""
        if not class_name:
            return []
        if seen is None:
            seen = set()
        cached = self._ancestor_cache.get(class_name) if not seen else None
        if cached is not None:
            return cached
        found: list = []
        for base in self.bases.get(class_name, ()):
            if base in seen or base == class_name:
                continue
            seen.add(base)
            found.append(base)
            found.extend(self._base_chain(base, seen))
        self._ancestor_cache.setdefault(class_name, found)
        return found

    def callable_alias(
        self,
        referenced: str,
        scope: str = "",
    ) -> str | None:
        """`runner = execute`: a reference to a first-party callable, not a call to one.

        Only a spelling that already resolves to something first-party is kept, so an
        ordinary constant assignment does not enter the table.
        """
        if not referenced:
            return None
        if self._local_function(referenced, scope) is not None:
            return referenced
        if self._targets(referenced.partition(".")[0]):
            return referenced
        return None

    def targets_of(
        self,
        callee: ast.AST,
        class_name: str = "",
        scope: str = "",
        instances: dict | None = None,
        aliases: dict | None = None,
    ) -> list:
        """Every first-party (file, qualname) a call can reach.

        This is what makes the analysis inter-procedural. Four forms matter: a bare local
        function, a method on a class defined here, a name imported with
        `from ... import f`, and a `module.f` where `module` was imported.

        A list, because one alias can be bound twice in the same file. Taking the first
        resolvable target meant that with `codec` bound to a clean parser in one function
        and a dirty one in another, the clean one was chosen for both and a value from the
        dirty parser reached a sink unreported. Every binding is analysed now, which is the
        same fail-closed choice the source and sink tables already make.
        """
        # Before the name guard: `super().execute(...)` reduces to a name the rest of this
        # cannot use, and on some shapes to nothing at all, so the early return fired and
        # taint never entered an inherited method. An overridden helper that passes its
        # argument to a sink was invisible from every subclass call site.
        if isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Call):
            if _call_name(callee.value.func).rpartition(".")[2] == "super" and class_name:
                # Through the shared resolver, so an inherited method defined in another
                # first-party module is reached. Checking this file only meant a tainted
                # argument handed to `super().execute(...)` never entered an imported
                # base, which is how most of this tree is laid out.
                inherited = self._inherited_methods_on(class_name, callee.attr)
                if inherited:
                    return inherited
                return []
        # `Child().execute(...)`: a method on a fresh instance. `_call_name` cannot
        # reduce a Call to a name, so the callee did not resolve at all and every method
        # reached this way, inherited ones included, sat outside the analysis.
        if isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Call):
            constructed = self.resolve_construction(_call_name(callee.value.func), scope)
            if constructed:
                return self._methods_on(constructed, callee.attr)
        name = _call_name(callee)
        if not name:
            return []
        # `runner = execute` then `runner(json.loads(blob))`. The alias is neither an
        # indexed local function named `runner` nor an import target, so the callee did
        # not resolve at all and taint never entered the helper. EVERY callable the name
        # was bound to, for the same reason reconstructed instance types keep all of
        # theirs: `runner = safe` then `runner = dirty` runs the second, and resolving
        # only the first left a sink inside it unreported. One substitution deep, so a
        # pair of names bound to each other cannot loop.
        candidates = list(aliases.get(name, ())) if aliases else []
        if not candidates:
            candidates = [name]
        found: list = []
        for candidate_name in candidates:
            for target in self._resolve_name(candidate_name, class_name, scope, instances):
                if target not in found:
                    found.append(target)
        return found

    def _resolve_name(self, name: str, class_name: str, scope: str, instances: dict | None) -> list:
        """One spelling, resolved. Split out so an alias bound twice resolves both."""
        head, _, tail = name.partition(".")
        # Bare call to a function defined in this file, nested helpers included.
        if not tail:
            local = self._local_function(name, scope)
            if local is not None:
                return [(self.path, local)]
        # `self.runner.execute(...)` on an instance held by an attribute. BEFORE the
        # `self.`/`cls.` branch below, which returns early on anything with that head
        # and so swallowed this shape: the longest dotted prefix that names a tracked
        # instance is the real receiver.
        if tail and instances and "." in name:
            prefix, _, attribute = name.rpartition(".")
            held = []
            for constructed in instances.get(prefix) or ():
                for target in self._methods_on(constructed, attribute):
                    if target not in held:
                        held.append(target)
            if held:
                return held
        # `self.method(...)` and `cls.method(...)`. Without this an instance method is
        # outside the analysis entirely: taint neither enters it nor returns from it, so
        # a class that parses an untrusted config in one method and dynamically imports
        # the result in another was accepted. Most code in this tree is methods.
        if head in ("self", "cls") and tail and class_name:
            # Through the shared resolver, so an inherited helper is found at any depth
            # and in another first-party file. Checking the current class and its direct
            # bases in this file only meant `self.execute(parsed)` in a child of an
            # imported base reached no target at all.
            return self._methods_on(class_name, tail)
        # `parser = Parser()` then `parser.parse(...)`. The head is a local variable, so
        # nothing resolved it and the method sat outside the analysis.
        if tail and instances:
            # Every class the name was constructed from, for the same fail-closed reason
            # the import table has: `runner = Safe()` then `runner = Dirty()` runs the
            # second, and resolving only the first left a sink inside it unreported.
            resolved = []
            for constructed in instances.get(head) or ():
                for target in self._methods_on(constructed, tail):
                    if target not in resolved:
                        resolved.append(target)
            if resolved:
                return resolved
        # `Runner(json.loads(blob)["command"])`: the call reaches `__init__`, which is
        # where the value is stored on the instance. Nothing resolved a class name used
        # as a callee, so a constructor that parks an argument on `self` and a method that
        # later executes it were both outside the analysis.
        if not tail:
            constructed = self.resolve_construction(name, scope)
            if constructed:
                local = f"{constructed}.__init__"
                if local in self.functions:
                    return [(self.path, local)]
                # The constructor has to actually exist before this branch claims the
                # call. `resolve_construction` resolves an imported NAME, not necessarily
                # a class, so accepting whatever the index returned sent every bare call
                # to an imported helper to a `f.__init__` that does not exist and the
                # import resolution below never ran: two dozen findings went quiet.
                file, module = self.index.resolve_module(constructed)
                if file is not None and module and constructed.startswith(module + "."):
                    inner = f"{constructed[len(module) + 1 :]}.__init__"
                    defining = self.index.facts.get(file)
                    if defining is not None and inner in defining.functions:
                        return [(file, inner)]
        # `from pkg.mod import f` then `f(...)`.
        targets = self._targets(head)
        if not targets:
            # `Parser.parse(...)` where Parser is a class defined in this file. The method
            # is already indexed as `Parser.parse`, but the head is neither an import nor
            # self, so the callee was rejected and a local static or class method sat
            # outside the analysis: it could return a parsed config straight into an
            # import_module with nothing reported.
            local = self._local_function(name, scope)
            if local is not None:
                return [(self.path, local)]
            return []
        imported: list = []
        for dotted in targets:
            full = f"{dotted}.{tail}" if tail else dotted
            file, module = self.index.resolve_module(full)
            if file is None:
                continue
            qualname = full[len(module) + 1 :] if module and full.startswith(module + ".") else ""
            hop = self.index.follow_reexport(file, qualname)
            if hop is not None:
                file, qualname = hop
            if (file, qualname) not in imported:
                imported.append((file, qualname))
        return imported


def _declared_here(node: ast.AST, kind) -> set:
    """`global`/`nonlocal` names declared by this scope, not by a nested one.

    A declaration inside a closure belongs to that closure, so collecting them with a
    plain `ast.walk` attributed an inner helper's binding to the function around it and
    turned a safe outer name into a tainted one.
    """
    found: set = set()

    def walk(current: ast.AST) -> None:
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, kind):
                found.update(child.names)
            walk(child)

    walk(node)
    return found


def _with(current, value: str) -> tuple:
    """`current` plus `value`, in first-seen order and without duplicates."""
    existing = current or ()
    return existing if value in existing else tuple(existing) + (value,)


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
        # local name -> class it was constructed from, for `parser = Parser()`
        self.instance_types: dict[str, tuple] = dict(facts.module_instances)
        # local name -> the sink it refers to, for `loader = importlib.import_module`
        # Every sink a name was bound to, not just the first: `action = subprocess.run`
        # then `action = sys.path.insert` runs the second, and the two watch different
        # argument positions, so keeping one examined the wrong index.
        self.sink_aliases: dict[str, tuple] = dict(facts.module_sink_aliases)
        # local name -> every first-party callable it refers to, for `runner = execute`
        self.callable_aliases: dict[str, tuple] = dict(facts.module_callable_aliases)
        # alias name -> how many positional arguments a `functools.partial` already
        # bound into it. `add_path = partial(sys.path.insert, 0)` shifts the path from
        # the sink's position 1 to the wrapper's position 0, so recording the sink
        # identity alone left the later `add_path(parsed)` compared against nothing.
        self.alias_offsets: dict[str, int] = dict(facts.module_alias_offsets)
        # Names bound to a dict whose KEYS are untrusted while every value is fixed.
        # Reading the keys has to stay tainted, since that is what iterating the dict
        # yields, but `.values()` hands back only the fixed half and reporting a sink on
        # it blocks code that is provably safe.
        self.key_taint_only: set = set()
        # local name -> the source it refers to, for `decode = json.loads`. Sink aliases
        # were followed and source aliases were not, so a deserialiser behind a local
        # name read clean and everything downstream of it did too.
        self.source_aliases: dict[str, str] = dict(facts.module_source_aliases)
        # Local names bound to a literal True, for `enabled = True` forwarded as
        # `trust_remote_code = enabled`.
        self.true_names: set[str] = set(facts.module_true_names)
        self.artefacts: set[str] = set()
        self.returns_tainted: str = ""
        self.findings: list[dict] = []
        # The WHOLE containing class, not the first segment. For `Outer.Runner.run`,
        # taking `Outer` sent `self.identity(...)` looking for `Outer.identity`, so a
        # method of a nested class resolved to nothing. Read off the recorded method set,
        # which knows whether the immediately enclosing scope is a class.
        self.class_name = qualname.rpartition(".")[0] if qualname in facts.methods else ""
        # Instances this class holds on attributes, so `self.runner.execute(...)` in one
        # method resolves against what `__init__` stored in another.
        if self.class_name:
            prefix = f"{facts.relative}::{self.class_name}."
            for key, types in sorted(state.attr_instances.items()):
                if not key.startswith(prefix):
                    continue
                spelling = f"self.{key[len(prefix) :]}"
                for candidate in types:
                    self.instance_types[spelling] = _with(
                        self.instance_types.get(spelling), candidate
                    )

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
            for key in self._attr_keys(node):
                attribute_reason = self.state.tainted_attrs.get(key)
                if attribute_reason:
                    return attribute_reason
            # `import producer` then `producer.MODEL_TYPE`. The earlier fix covered only
            # `from producer import MODEL_TYPE`, so the module-qualified spelling of the
            # same tainted global fell through to the clean base name.
            if isinstance(node.value, ast.Name):
                module_reason = self._module_global(node.value.id, node.attr)
                if module_reason:
                    return module_reason
            method = _matches(_call_name(node), UNTRUSTED_METHODS)
            if method:
                return f"read via .{method}"
            # `cfg.module` where `module` is an `@property` whose getter returns a parsed
            # value. An ordinary method call on the same instance was followed, so the
            # one spelling that looks like an attribute read was the gap.
            property_reason = self._property_reason(node)
            if property_reason:
                return property_reason
            return self.tainted(node.value)
        if isinstance(node, ast.Subscript):
            # `_DTYPES[meta["dtype"]]` is the raising form of the same translation table
            # lookup, and the result is one of the literals in it either way. Reading it
            # through the base would have carried the key's taint into a constant.
            if _call_name(node.value) in self.facts.constant_maps:
                return None
            return self.tainted(node.value)
        if isinstance(node, ast.Starred):
            return self.tainted(node.value)
        if isinstance(node, ast.NamedExpr):
            # `import_module((name := parsed["module"]))`. The visitor binds `name` only
            # after the call's sink check has run, and there was no case for the
            # expression itself, so the value passed to the sink read clean on every
            # pass of the fixpoint.
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
            # Keys as well as values: iterating a dict yields its keys, so
            # `{parsed["module"]: None}` exposed the untrusted name directly while the
            # container read clean. `_dict_halves` keeps the two apart for the one
            # consumer that can tell them apart.
            keys, values = self._dict_halves(node)
            return values or keys
        return None

    def _dict_halves(self, node: ast.Dict) -> tuple:
        """`(reason from any key, reason from any value)` for a dict literal."""
        keys = None
        for key in node.keys:
            if key is None:
                # `{**other}` carries whatever the other mapping holds, on both halves.
                continue
            keys = keys or self.tainted(key)
        values = None
        for value in node.values:
            values = values or self.tainted(value)
        for index, key in enumerate(node.keys):
            if key is None:
                spread = self.tainted(node.values[index])
                keys = keys or spread
                values = values or spread
        return (keys, values)

    def _tainted_call(self, node: ast.Call) -> str | None:
        names = self.facts.canonicals(_call_name(node.func))
        source = _matches_any(names, UNTRUSTED_CALLS)
        if source is None:
            # `decode = json.loads` then `decode(blob)`. Sink aliases were followed and
            # source aliases were not, so a deserialiser behind an ordinary local name
            # read clean and everything downstream of it did too.
            source = self.source_aliases.get(_call_name(node.func))
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
        # A lookup in a literal translation table, whatever the key. Before the
        # passthrough list below, which would otherwise carry the key's taint into the
        # looked-up constant.
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ("get", "pop")
            and _call_name(node.func.value) in self.facts.constant_maps
        ):
            return None
        # `.values()` on a dict whose keys alone are untrusted hands back the fixed
        # half, and treating the whole literal as tainted blocked provably safe code.
        # `.keys()` and `.items()` both expose the key, so only this one is narrowed.
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "values"
            and _call_name(node.func.value) in self.key_taint_only
        ):
            return None
        # Container and string operations preserve taint.
        if isinstance(node.func, ast.Attribute) and node.func.attr in (
            "get",
            "pop",
            "format",
            # The mapping spelling of the same interpolation.
            "format_map",
            "join",
            "split",
            "strip",
            # Trimming a known prefix or suffix off an untrusted name selects a
            # different module, it does not validate anything:
            # `parsed["module"].removeprefix("plugins.")` reached an import clean.
            "removeprefix",
            "removesuffix",
            "lstrip",
            "rstrip",
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
            # `urlopen(url).read().decode().strip()`: without decode the bytes read clean
            # the moment they became a str, and everything chained after it inherited that.
            "decode",
            "encode",
            # A defensive copy is not a sanitiser. `cfg.copy()` read clean, so an ordinary
            # container copy laundered a parsed config on its way to a sink.
            "copy",
        ):
            reason = self.tainted(node.func.value)
            if reason:
                return reason
            # Keywords too: `"x.{n}".format(n = parsed)` is the named spelling of the
            # same interpolation, and checking positions only returned a clean string.
            for argument in list(node.args) + [k.value for k in node.keywords]:
                reason = self.tainted(argument)
                if reason:
                    return reason
            return None
        # Container constructors pass their contents straight through, so wrapping a
        # tainted iterable in one of these is not a sanitiser.
        if _matches_any(
            names,
            {
                "list",
                "tuple",
                "set",
                "dict",
                "sorted",
                "reversed",
                "iter",
                "enumerate",
                "copy.copy",
                "copy.deepcopy",
                # `next(parse(blob))` is how a generator's first value is taken, and
                # without it the return summary the yield fix produces was dropped again
                # at the consumer.
                "next",
                # `map(str, parsed["modules"])` does not validate anything: the names
                # come straight out of the artefact and the loop over the result read
                # clean. `zip` is the same shape with two iterables.
                "map",
                "filter",
                "zip",
            },
        ):
            # Keywords too: `copy.deepcopy(x = json.loads(blob))` and `dict(**...)` style
            # calls carry their input by name, and checking positions only handed back a
            # clean value.
            for argument in list(node.args) + [k.value for k in node.keywords]:
                reason = self.tainted(argument)
                if reason:
                    return reason
            return None
        # `open(downloaded)` hands back a handle onto attacker bytes, and the usual shape
        # is `with open(downloaded, "rb") as handle: pickle.load(handle)`. Without this
        # the handle read clean and the declared pickle sink never fired.
        if _matches_any(names, {"open", "io.open", "Path.open"}):
            # Keywords too: open takes its path as `file=`, and checking only positions
            # left `open(file = downloaded, mode = "rb")` handing back a clean handle.
            arguments = list(node.args) + [k.value for k in node.keywords]
            # And the receiver, because `Path(downloaded).open("rb")` carries the path
            # there rather than in any argument: the arguments are the mode, and the
            # handle onto attacker bytes read clean for the whole instance-method form.
            if isinstance(node.func, ast.Attribute) and node.func.attr == "open":
                arguments.append(node.func.value)
            for argument in arguments:
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
        # A first-party callee that returns tainted data. Any of them: an alias bound
        # twice resolves to more than one callee, and only one of them need be dirty.
        for target in self.facts.targets_of(
            node.func,
            self.class_name,
            scope = self.qualname,
            instances = self.instance_types,
            aliases = self.callable_aliases,
        ):
            returned = self.state.returns_tainted.get(target)
            if returned:
                return returned
        return None

    def _property_reason(self, node: ast.Attribute) -> str | None:
        """The return summary of an `@property` getter reached through an instance.

        The receiver does not have to be a plain name. `self.cfg = Config()` is tracked
        as a held instance, yet requiring an `ast.Name` here meant
        `import_module(self.cfg.module)` could not consult `Config.module`'s summary and
        was accepted while the getter returns a parsed value.
        """
        if isinstance(node.value, ast.Call):
            # `Config().module`: the receiver is the construction itself, which has no
            # name at all, so the lookup gave up before it started.
            constructed = self.facts.resolve_construction(
                _call_name(node.value.func), self.qualname
            )
            return self._property_on(node.attr, (constructed,) if constructed else ())
        holder = node.value.id if isinstance(node.value, ast.Name) else _call_name(node.value)
        if not holder:
            return None
        if holder in ("self", "cls"):
            owners = tuple((self.class_name,) if self.class_name else ())
        else:
            owners = tuple(self.instance_types.get(holder) or ())
            if not owners and isinstance(node.value, ast.Attribute):
                # Held on an attribute rather than a local, which is where a constructor
                # puts it, so the type lives in the shared map the attribute keys use.
                for key in self._attr_keys(node.value):
                    owners = owners + tuple(self.state.attr_instances.get(key) or ())
        return self._property_on(node.attr, owners)

    def _property_on(self, attribute: str, owners) -> str | None:
        """The getter's return summary, for any of `owners`."""
        for owner in owners:
            for file, qualname in self.facts._methods_on(owner, attribute):
                if qualname not in (self.facts.index.facts.get(file) or self.facts).properties:
                    continue
                returned = self.state.returns_tainted.get((file, qualname))
                if returned:
                    return returned
        return None

    def _module_global(self, alias: str, attribute: str) -> str | None:
        """`producer.MODEL_TYPE`, where `producer` is an imported first-party module."""
        for dotted in self.facts._targets(alias):
            file = self.facts.index.resolve(dotted)
            if file is None:
                continue
            reason = self.state.tainted_globals.get(f"{_relative(file)}::{attribute}")
            if reason:
                return reason
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

    def _attr_keys(self, node: ast.Attribute) -> list:
        """Every key `node` can mean: each possible owner class and its ancestors.

        `child.command = parsed` recorded `Child.command` while an inherited `Base.run`
        reads `self.command` as `Base.command`, so the write and the read never met and
        the sink inside the inherited method was missed. Writes bind all of them and
        reads consult all of them, which is the same fail-closed choice the rest of the
        resolution makes.
        """
        keys: list = []
        seen: set = set()
        for facts, class_name in self._attr_owners(node):
            self._collect_attr_keys(facts, class_name, node.attr, keys, seen)
        return keys

    def _collect_attr_keys(self, facts, class_name: str, attribute: str, keys, seen) -> None:
        """`class_name` and every ancestor of it, each keyed against its OWN file.

        Resolved in the file that declares each base rather than in the writer's: a
        local child of an imported middle class whose root is declared somewhere else
        again had the root's key written against the middle's file, so the inherited
        reader never saw it.
        """
        marker = (str(facts.path), class_name)
        if not class_name or marker in seen:
            return
        seen.add(marker)
        key = f"{facts.relative}::{class_name}.{attribute}"
        if key not in keys:
            keys.append(key)
        for base in facts.bases.get(class_name, ()):
            if base in facts.classes:
                self._collect_attr_keys(facts, base, attribute, keys, seen)
            for dotted in facts._targets(base):
                file, module = facts.index.resolve_module(dotted)
                if file is None or not module or not dotted.startswith(module + "."):
                    continue
                inner = dotted[len(module) + 1 :]
                defining = facts.index.facts.get(file)
                if defining is None:
                    candidate = f"{_relative(file)}::{inner}.{attribute}"
                    if candidate not in keys:
                        keys.append(candidate)
                    continue
                self._collect_attr_keys(defining, inner, attribute, keys, seen)

    def _attr_owners(self, node: ast.Attribute):
        """`(facts, class name)` for every class the receiver can be.

        EVERY recorded type, not the first: `runner = Safe()` then `runner = Dirty()`
        resolves its method call against both, so recording the write against `Safe`
        alone meant a sink in `Dirty.run` reading `self.command` saw a clean value. The
        earlier note calling that an accepted margin no longer applies now that the
        caller takes a list.
        """
        if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
            # `cls` as well as `self`: a classmethod writing `cls.command` and another
            # reading it got an empty key on both halves.
            return [(self.facts, self.class_name)] if self.class_name else []
        spelling = node.value.id if isinstance(node.value, ast.Name) else _call_name(node.value)
        found: list = []
        # `Config.module` reads the class attribute directly, so the receiver is the
        # class and not an instance of one. Only instances were resolved, so a class
        # body's binding had nowhere to be read from under this spelling.
        if spelling in self.facts.classes:
            found.append((self.facts, spelling))
        for constructed in self.instance_types.get(spelling) or ():
            if constructed in self.facts.classes:
                if (self.facts, constructed) not in found:
                    found.append((self.facts, constructed))
                continue
            # An imported class belongs to the file that declares it, because that is
            # the file whose `self.attr` reads have to see this.
            file, module = self.facts.index.resolve_module(constructed)
            if file is None or not module or not constructed.startswith(module + "."):
                continue
            owner = constructed[len(module) + 1 :]
            defining = self.facts.index.facts.get(file)
            if owner and defining is not None and (defining, owner) not in found:
                found.append((defining, owner))
        return found

    def _attr_key(self, node: ast.Attribute) -> str:
        """The first key, for the one caller that wants a single spelling."""
        keys = self._attr_keys(node)
        return keys[0] if keys else ""

    def _ancestors(
        self,
        class_name: str,
        seen: set | None = None,
    ) -> list:
        """Declared base classes of `class_name`, transitively, within this file."""
        return self.facts._base_chain(class_name, seen)

    def _attr_key(self, node: ast.Attribute) -> str:
        """The first key, for the one caller that wants a single spelling."""
        owners = self._attr_owners(node)
        return owners[0] if owners else ""

    # -- taint writes ------------------------------------------------------------------

    def _assign(
        self,
        target: ast.AST,
        reason: str,
        value: ast.AST | None = None,
    ) -> None:
        """`value` is the expression assigned, when the caller has it.

        Only one check needs it, and it needs the expression rather than the reason: a
        dict whose keys alone are untrusted is tainted, yet its `.values()` is not.
        """
        if isinstance(target, ast.Name):
            # Flow-insensitive, and the strongest reason wins: a name tainted by a real
            # read stays tier A even if it is also assigned from a named parameter.
            if self.local_reasons.get(target.id, NAMED_PARAM_REASON) == NAMED_PARAM_REASON:
                self.local_reasons[target.id] = reason
            # `table = {parsed["label"]: ["echo", "ok"]}`: the keys carry it and the
            # values do not, so `table.values()` is every bit as fixed as the literal.
            # Any other assignment to the same name clears the distinction.
            if isinstance(value, ast.Dict):
                keys, values = self._dict_halves(value)
                if keys and not values:
                    self.key_taint_only.add(target.id)
                else:
                    self.key_taint_only.discard(target.id)
            else:
                self.key_taint_only.discard(target.id)
            # `global MODEL_TYPE` then assigning it is a write to module state, and only
            # module-level statements used to produce one. A function that parsed a config
            # into a declared global and another that read it at runtime were each clean.
            if target.id in self.facts.globals_declared.get(self.qualname, ()):
                self._bind(
                    self.state.tainted_globals, f"{self.facts.relative}::{target.id}", reason
                )
        elif isinstance(target, ast.Attribute):
            for key in self._attr_keys(target):
                # Same tier-A-wins rule as locals and parameters. Writing unconditionally
                # let one method's tier-B assignment to self.command overwrite another
                # method's tier-A one, purely on the order the methods are visited, and
                # the sink reading that attribute then did not gate.
                self._bind(self.state.pending_attrs, key, reason)
        elif isinstance(target, ast.Subscript):
            # `settings["module"] = json.loads(blob)`. The container is what carries it
            # from here, and `tainted()` already reads a Subscript through its base, so
            # tainting the base is both conservative and the shape the reader expects.
            # Ignoring these targets let a config assembled key by key arrive clean.
            base = target.value
            # `full_state.setdefault(path, {})[name] = tensor` writes THROUGH a call, so
            # the base is a Call and the assignment landed nowhere: the dict being built
            # stayed clean and the function returning it was summarised clean too. The
            # receiver of the call is the container that keeps the value.
            if isinstance(base, ast.Call) and isinstance(base.func, ast.Attribute):
                if base.func.attr in self._MUTATORS or base.func.attr in ("get", "pop"):
                    base = base.func.value
            self._assign(base, reason)
        elif isinstance(target, (ast.Tuple, ast.List)):
            # Unpacking, so the element is not the dict itself and the key-only
            # distinction does not survive it.
            for element in target.elts:
                self._assign(element, reason)

    def visit_Assign(self, node: ast.Assign) -> None:
        reason = self.tainted(node.value)
        if reason:
            for target in node.targets:
                self._assign(target, reason, node.value)
        self._note_true(node)
        self._note_construction(node)
        self._note_sink_alias(node)
        self._note_source_alias(node)
        self._note_instance_alias(node)
        self._note_callable_alias(node)
        self.generic_visit(node)

    def _note_sink_alias(self, node: ast.Assign) -> None:
        """`loader = importlib.import_module`: a reference to a sink, not a call to one."""
        if isinstance(node.value, ast.Call):
            # `runner = functools.partial(subprocess.run)` is still a reference to the
            # sink, wrapped. The call guard above skipped it, so the partial walked past
            # the gate while the bare alias of the same sink was caught.
            wrapped = _matches_any(
                self.facts.canonicals(_call_name(node.value.func)),
                {"functools.partial", "partial"},
            )
            if wrapped is None or not node.value.args:
                return
            inner = _call_name(node.value.args[0])
            # The partial's own call with the wrapped callee dropped off the front, so
            # the positions line up with whatever it wraps. Used for the creation-time
            # check and for the pre-bound arguments of a first-party callee alike.
            bound = ast.Call(
                func = node.value.args[0],
                args = list(node.value.args[1:]),
                keywords = list(node.value.keywords),
            )
            ast.copy_location(bound, node.value)
            # How many positional arguments the wrapper no longer takes. Every later
            # call through the alias is shifted left by this much.
            # The wrapper's own arguments PLUS whatever the thing it wraps had already
            # bound: `invoke = partial(add_path)` around an existing partial recorded no
            # offset at all, so the call was checked at the original sink position.
            prebound = len(node.value.args) - 1 + self.alias_offsets.get(inner, 0)
            direct = _matches_any(self.facts.canonicals(inner), SINKS)
            # Every sink the inner name can be, not the first: after `action =
            # subprocess.run` then `action = sys.path.insert`, keeping only the first
            # meant the partial was checked against a sink whose watched position the
            # offset had already removed, and the one it really calls was never checked.
            # Only real table entries: a partial of a `getattr` or `torch.load` alias
            # would otherwise index SINKS with a marker that is not a key.
            candidates_held = (
                [direct]
                if direct is not None
                else [
                    candidate
                    for candidate in self.sink_aliases.get(inner) or ()
                    if candidate in SINKS
                ]
            )
            held = candidates_held[0] if candidates_held else None
            if held is None:
                # Not a sink, but `partial` wraps first-party helpers at least as often:
                # `runner = partial(execute)` with `execute` passing its parameter to
                # `subprocess.run` recorded nothing at all, so neither the alias nor the
                # pre-bound arguments ever reached the helper's parameters.
                relayed = self.callable_aliases.get(inner) or ()
                candidates = list(relayed) or (
                    [self.facts.callable_alias(inner, self.qualname)]
                    if self.facts.callable_alias(inner, self.qualname)
                    else []
                )
                if not candidates:
                    return
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    for candidate in candidates:
                        self.callable_aliases[target.id] = _with(
                            self.callable_aliases.get(target.id), candidate
                        )
                    if prebound:
                        self.alias_offsets[target.id] = prebound
                # The arguments bound at creation reach the helper's parameters on every
                # later call, exactly as the wrapped-sink case checks them against the
                # sink.
                if node.value.args[1:] or node.value.keywords:
                    self._propagate_into_callee(bound)
                return
            for target in node.targets:
                spelling = self._alias_target(target)
                for candidate in candidates_held:
                    if spelling:
                        self.sink_aliases[spelling] = _with(
                            self.sink_aliases.get(spelling), candidate
                        )
                    self._note_attr_sink_alias(target, candidate)
                if spelling and prebound:
                    self.alias_offsets[spelling] = prebound
            # The arguments a partial pre-binds reach the sink whenever the wrapper is
            # called, and recording only the sink's identity discarded them: a later
            # `loader()` leaves the sink check nothing to look at.
            for candidate in candidates_held:
                self._check_one_sink(bound, candidate)
            return
        referenced = _call_name(node.value)
        if not referenced:
            return
        # `loader = importlib.import_module` then `invoke = loader`. The second assignment
        # does not canonically match a sink, so it was discarded and the second alias
        # walked past the gate. One hop per pass, and the fixpoint settles the chain.
        chained = self.sink_aliases.get(referenced)
        if chained:
            carried = self.alias_offsets.get(referenced)
            for target in node.targets:
                spelling = self._alias_target(target)
                if spelling:
                    for candidate in chained:
                        self.sink_aliases[spelling] = _with(
                            self.sink_aliases.get(spelling), candidate
                        )
                    # `invoke = add_path` keeps the partial's layout: without this the
                    # second name was checked at the sink's own positions again.
                    if carried:
                        self.alias_offsets[spelling] = carried
            return
        sink = _matches_any(self.facts.canonicals(referenced), SINKS)
        if sink is None:
            sink = _matches_any(self.facts.canonicals(referenced), {"getattr"})
            if sink is not None:
                sink = "getattr"
        if sink is None and (
            _matches_any(self.facts.canonicals(referenced), {"torch.load"}) is not None
        ):
            # `loader = torch.load` then `loader(downloaded, weights_only = False)`.
            sink = TORCH_LOAD_ALIAS
        if sink is None:
            return
        for target in node.targets:
            spelling = self._alias_target(target)
            if spelling:
                self.sink_aliases[spelling] = _with(self.sink_aliases.get(spelling), sink)
            self._note_attr_sink_alias(target, sink)

    def _note_attr_sink_alias(self, target: ast.AST, sink: str) -> None:
        """`self.loader = importlib.import_module` read back from a different method.

        The local spelling covers one function. A constructor storing the sink and a
        method calling through it are separate passes, so the binding goes into the
        shared map under the same keys the attribute reads use.
        """
        if not isinstance(target, ast.Attribute):
            return
        for key in self._attr_keys(target):
            self.state.attr_sink_aliases[key] = _with(self.state.attr_sink_aliases.get(key), sink)

    @staticmethod
    def _alias_target(target: ast.AST) -> str:
        """The spelling an alias is stored under, for a name or an attribute target.

        `runner.loader = importlib.import_module` was discarded because only names were
        accepted, and `runner.loader(parsed)` then matched neither the canonical table
        nor the alias table, so the dynamic import went through. The call site spells the
        receiver out, so storing it under that spelling is what the lookup already asks
        for.
        """
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute):
            return _call_name(target)
        return ""

    def seed_defaults(self) -> None:
        """`def execute(argv = CONFIG["argv"])` runs the default when a caller omits it.

        A parameter was tainted only by an explicit caller or by its name, so a call with
        the argument left out evaluated a default reading a parsed global while the
        parameter itself stayed clean, and the sink below it was reported nowhere.
        """
        node = self.facts.functions.get(self.qualname)
        if node is None:
            return
        arguments = node.args
        positional = list(arguments.posonlyargs) + list(arguments.args)
        supplied = arguments.defaults
        pairs = (
            list(zip(positional[len(positional) - len(supplied) :], supplied)) if supplied else []
        )
        pairs += [
            (parameter, default)
            for parameter, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
            if default is not None
        ]
        for parameter, default in pairs:
            reason = self.tainted(default)
            if reason:
                self._bind(self.local_reasons, parameter.arg, reason)

    def _note_true(self, node: ast.Assign) -> None:
        """`enabled = True`, in either the plain or the annotated spelling.

        An alias of a known-true name counts as well: `enabled = True` then
        `remote = enabled` reaches the loader as True, and one ordinary assignment was
        enough to walk past the gate.
        """
        literal = isinstance(node.value, ast.Constant) and node.value.value is True
        aliased = isinstance(node.value, ast.Name) and node.value.id in self.true_names
        if not (literal or aliased):
            return
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.true_names.add(target.id)

    def _note_source_alias(self, node: ast.Assign) -> None:
        """`decode = json.loads`: a reference to a source, not a call to one."""
        if isinstance(node.value, ast.Call):
            return
        referenced = _call_name(node.value)
        if not referenced:
            return
        source = self.source_aliases.get(referenced) or _matches_any(
            self.facts.canonicals(referenced), UNTRUSTED_CALLS
        )
        if source is None:
            return
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.source_aliases.setdefault(target.id, source)

    def _note_instance_alias(self, node: ast.Assign) -> None:
        """`invoke = runner`: the second name is the same object as the first.

        Without this the copy carried no type, so `invoke.execute(parsed)` resolved to
        nothing while `runner.execute(parsed)` resolved.
        """
        if isinstance(node.value, ast.Call):
            return
        referenced = _call_name(node.value)
        held = self.instance_types.get(referenced)
        if not held:
            return
        for target in node.targets:
            if isinstance(target, ast.Name):
                for candidate in held:
                    self.instance_types[target.id] = _with(
                        self.instance_types.get(target.id), candidate
                    )

    def _note_callable_alias(self, node: ast.Assign) -> None:
        """`runner = execute`: a reference to a first-party helper, not a call to one."""
        if isinstance(node.value, ast.Call):
            return
        referenced = _call_name(node.value)
        # `execute = runner.execute` saves a BOUND method. The prefix names a tracked
        # instance, so the spelling resolves through the instance branch, but
        # `callable_alias` rejected it as neither a local function nor an import target.
        if "." in referenced and self.instance_types.get(referenced.rpartition(".")[0]):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.callable_aliases[target.id] = _with(
                        self.callable_aliases.get(target.id), referenced
                    )
            return
        if self.sink_aliases.get(referenced):
            return
        # `runner = execute` then `invoke = runner`: the second name refers to the same
        # helper, and `callable_alias` rejects `runner` because it is not itself an
        # indexed function or an import target, so taint stopped at the first hop.
        inherited = self.callable_aliases.get(referenced)
        if inherited:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    for candidate in inherited:
                        self.callable_aliases[target.id] = _with(
                            self.callable_aliases.get(target.id), candidate
                        )
            return
        alias = self.facts.callable_alias(referenced, self.qualname)
        if not alias:
            return
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.callable_aliases[target.id] = _with(
                    self.callable_aliases.get(target.id), alias
                )

    def _note_construction(self, node: ast.Assign) -> None:
        """`parser = Parser()`, so `parser.parse(...)` resolves to `Parser.parse`.

        Without this the head is neither an import nor a class name and the callee did not
        resolve at all, so a method that returns a parsed config fed an import_module with
        nothing reported. Only classes defined in this file, and only a direct call, which
        is the shape that can be read off the source with no inference.
        """
        if not isinstance(node.value, ast.Call):
            return
        # Including `from producer import Parser` then `parser = Parser()`: an imported
        # first-party class resolves to its dotted target, so its methods are inside the
        # analysis exactly as a locally declared class's are.
        constructed = self.facts.resolve_construction(_call_name(node.value.func), self.qualname)
        if not constructed:
            return
        for target in node.targets:
            # `self.runner = Runner()` is how composition is written, and only plain
            # locals were recorded, so `self.runner.execute(parsed)` resolved to nothing.
            # Keyed by the dotted text, which is what the call site spells.
            if isinstance(target, ast.Name):
                self.instance_types[target.id] = _with(
                    self.instance_types.get(target.id), constructed
                )
                continue
            if not isinstance(target, ast.Attribute):
                continue
            # `self.runner = Runner()` is how composition is written. Recorded in shared
            # state under the same key the attribute reads use, because the constructor
            # stores it and another method calls through it.
            spelling = _call_name(target)
            if spelling:
                self.instance_types[spelling] = _with(
                    self.instance_types.get(spelling), constructed
                )
            for key in self._attr_keys(target):
                self.state.attr_instances[key] = _with(
                    self.state.attr_instances.get(key), constructed
                )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            reason = self.tainted(node.value)
            if reason:
                self._assign(node.target, reason, node.value)
            # `parser: Parser = Parser()` is the same construction as the plain form, and
            # only the plain form was recording it, so the annotated spelling left the
            # method unresolvable.
            synthetic = ast.Assign(targets = [node.target], value = node.value)
            # `enabled: bool = True` is the same flag as the plain form, and this mirrored
            # handling omitted the update that `visit_Assign` performs.
            self._note_true(synthetic)
            self._note_construction(synthetic)
            self._note_sink_alias(synthetic)
            self._note_source_alias(synthetic)
            # `invoke: Runner = runner` keeps the type, and this mirrored path omitted
            # the one helper that records it, so the method behind the second name did
            # not resolve while the unannotated spelling did.
            self._note_instance_alias(synthetic)
            self._note_callable_alias(synthetic)
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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_nested(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_nested(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_nested(node)

    def _visit_nested(self, node: ast.AST) -> None:
        """Scan a nested body for sinks, then drop the names it bound.

        The nested body is still walked, because a closure reading a tainted enclosing
        local is a real flow and that read has to be seen. What was wrong was keeping the
        names it WROTE: a nested helper assigning `command = json.loads(blob)` tainted the
        enclosing `command` as well, so a later `subprocess.run(command)` on the outer
        name was reported even though it only ever held a fixed value. The nested body is
        also indexed and analysed under its own qualname, so nothing is lost by scoping
        its writes to itself.
        """
        saved = (
            dict(self.local_reasons),
            dict(self.instance_types),
            dict(self.sink_aliases),
            dict(self.callable_aliases),
            dict(self.source_aliases),
            set(self.true_names),
            dict(self.alias_offsets),
            set(self.key_taint_only),
        )
        # A nested parameter SHADOWS the enclosing name for the whole nested body, and
        # the walk kept the enclosing binding live, so an outer tainted `command` made
        # `def helper(command): subprocess.run(command)` report even when the helper is
        # only ever called with a literal. Masked here, which costs nothing: the nested
        # body is analysed under its own qualname with its parameters bound by its real
        # callers, so a caller that does pass a tainted value is still reported there.
        for name in _param_names(node) if not isinstance(node, ast.ClassDef) else ():
            self.local_reasons.pop(name, None)
            self.instance_types.pop(name, None)
            self.sink_aliases.pop(name, None)
            self.callable_aliases.pop(name, None)
            self.source_aliases.pop(name, None)
            self.alias_offsets.pop(name, None)
            self.true_names.discard(name)
            self.key_taint_only.discard(name)
        # The nested body's `return` is the nested function's output, not this one's.
        # Leaving it set made an outer function that returns a fixed literal carry a
        # nested helper's tainted summary, and every caller of the outer one then failed
        # the gate on a value it never produces.
        returned = self.returns_tainted
        self.generic_visit(node)
        self.returns_tainted = returned
        # A `nonlocal` write is a write to THIS scope, which is the whole point of the
        # declaration, so dropping it with the nested body's own names discarded a real
        # flow: a helper setting an outer `command` from a parsed config left the sink
        # below it clean. Only ordinary nested locals are dropped.
        # The immediate nested scope only. A `nonlocal` inside a deeper closure targets
        # ITS enclosing function, not this one, so collecting them all propagated a
        # grandchild's write two levels up and rejected a safe sink here.
        shared = _declared_here(node, ast.Nonlocal)
        reasons = dict(saved[0])
        for name in sorted(shared):
            carried = self.local_reasons.get(name)
            if carried and reasons.get(name, NAMED_PARAM_REASON) == NAMED_PARAM_REASON:
                reasons[name] = carried
        # A `nonlocal` rebinding is a write to THIS scope whatever it writes, so the
        # alias tables have to follow the taint reason out: `nonlocal action; action =
        # subprocess.run` in a helper left the outer `action(parsed)` matching nothing at
        # all. Only the declared names, and only where this scope has no stronger entry.
        self.local_reasons = reasons
        self.instance_types = self._carry_shared(saved[1], self.instance_types, shared)
        self.sink_aliases = self._carry_shared(saved[2], self.sink_aliases, shared)
        self.callable_aliases = self._carry_shared(saved[3], self.callable_aliases, shared)
        # The source table too: a nested helper binding `decode = json.loads` made the
        # OUTER function's own safe `decode` read as a deserialiser, so a benign call
        # was reported even when the nested helper is never invoked.
        self.source_aliases = self._carry_shared(saved[4], self.source_aliases, shared)
        # And the true-flag set: a nested `enabled = True` leaked outward and made an
        # outer loader call that always receives False report a gated finding.
        self.true_names = saved[5] | {name for name in shared if name in self.true_names}
        # And the partial layouts, for the same reason: a nested wrapper's offset
        # applied to an outer name that was never wrapped.
        self.alias_offsets = self._carry_shared(saved[6], self.alias_offsets, shared)
        # The key-only set is the one piece of state here that SUPPRESSES a finding, so
        # it is the one that must not leak either way: a nested `table = {parsed: fixed}`
        # would otherwise clear an outer `table` that really does hold parsed values.
        # It is also deliberately not carried between passes of the fixpoint, because
        # accumulating a suppressor across passes can only lose a real finding.
        self.key_taint_only = saved[7]

    @staticmethod
    def _carry_shared(outer: dict, nested: dict, shared: set) -> dict:
        """`outer`, plus whatever the nested body bound for a `nonlocal` name."""
        restored = dict(outer)
        for name in sorted(shared):
            bound = nested.get(name)
            if bound is not None and name not in restored:
                restored[name] = bound
        return restored

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def _visit_comprehension(self, node: ast.AST) -> None:
        """Bind each `for` target before the element is walked.

        `[import_module(name) for name in json.loads(blob)["modules"]]` is one of the most
        common shapes in this tree, and the target was never bound, so the sink inside the
        element read `name` as clean. Taking the whole comprehension's value was already
        handled; it was the sink INSIDE one that was invisible.
        """
        # Python 3 gives a comprehension its own scope, so the target does not overwrite
        # a same-named variable around it. Keeping the binding afterwards turned a safe
        # outer `command` into a tainted one and blocked a sink that only ever sees the
        # fixed value.
        saved = dict(self.local_reasons)
        for generator in node.generators:
            reason = self.tainted(generator.iter)
            if reason:
                self._assign(generator.target, reason)
        self.generic_visit(node)
        bound = {
            name
            for generator in node.generators
            for target in ast.walk(generator.target)
            if isinstance(target, ast.Name)
            for name in [target.id]
        }
        for name in bound:
            if name in saved:
                self.local_reasons[name] = saved[name]
            else:
                self.local_reasons.pop(name, None)

    def visit_Match(self, node: ast.Match) -> None:
        """`case {"module": name}` binds `name` out of the subject.

        Structural matching is how a parsed config gets destructured, and the captures
        were never bound, so everything a `match` pulled out of attacker JSON read clean.
        """
        reason = self.tainted(node.subject)
        if reason:
            for case in node.cases:
                for captured in ast.walk(case.pattern):
                    name = getattr(captured, "name", None)
                    if isinstance(name, str):
                        self._bind(self.local_reasons, name, reason)
                    rest = getattr(captured, "rest", None)
                    if isinstance(rest, str):
                        self._bind(self.local_reasons, rest, reason)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        # The async form fell through to generic traversal, so `async for name in
        # parse(blob)` left the loop target clean and an async consumer of a parsed
        # config bypassed the gate entirely.
        self.visit_For(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                reason = self.tainted(item.context_expr)
                if reason:
                    self._assign(item.optional_vars, reason)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        # The async form fell through to generic traversal, so `async with
        # downloaded(repo) as path` left `path` clean and the sink below it was reported
        # nowhere. Same binding as the synchronous form.
        self.visit_With(node)

    def visit_Return(self, node: ast.Return) -> None:
        self._note_output(node.value)
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        # A generator's output is what it yields. Only Return was handled, so a helper
        # that iterates a parsed config and yields each name had no tainted summary and
        # every caller looping over it read clean.
        self._note_output(node.value)
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._note_output(node.value)
        self.generic_visit(node)

    def _note_output(self, value) -> None:
        if value is None:
            return
        reason = self.tainted(value)
        if reason and (not self.returns_tainted or self.returns_tainted == NAMED_PARAM_REASON):
            self.returns_tainted = reason

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            lowered = node.value.lower()
            for artefact in UNTRUSTED_ARTEFACTS:
                if artefact in lowered:
                    self.artefacts.add(artefact)
        self.generic_visit(node)

    # -- sinks -------------------------------------------------------------------------

    # Methods that put their argument inside the receiver. `settings.update(parsed)` is
    # an assignment into the container written as a call, and only assignments were
    # handled, so a config filled in this way arrived clean at the sink.
    _MUTATORS = frozenset({"update", "extend", "append", "add", "insert", "setdefault"})

    def visit_Call(self, node: ast.Call) -> None:
        self._note_mutation(node)
        self._propagate_into_callee(node)
        self._propagate_into_mapped(node)
        self._check_sink(node)
        self._check_torch_load(node)
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

    def _note_mutation(self, node: ast.Call) -> None:
        """`settings.update(json.loads(blob))` taints `settings`."""
        if not isinstance(node.func, ast.Attribute):
            return
        if node.func.attr not in self._MUTATORS:
            return
        for argument in list(node.args) + [k.value for k in node.keywords]:
            reason = self.tainted(argument)
            if reason:
                self._assign(node.func.value, reason)
                return

    def _propagate_into_callee(self, node: ast.Call) -> None:
        """Taint the callee's parameters, which is how a chain crosses a file."""
        for target in self.facts.targets_of(
            node.func,
            self.class_name,
            scope = self.qualname,
            instances = self.instance_types,
            aliases = self.callable_aliases,
        ):
            self._propagate_into_one(node, target)

    def _propagate_into_one(self, node: ast.Call, target) -> None:
        params = self.state.params.get(target)
        if not params:
            return
        bound = self.state.pending_params.setdefault(target, {})
        # `Runner(command)` is spelled with no receiver at all, yet `__init__` still
        # takes `self` first, so without the offset the argument bound to `self` and the
        # real parameter stayed clean.
        constructing = target[1].rpartition(".")[2] == "__init__" and not isinstance(
            node.func, ast.Attribute
        )
        # `execute = runner.execute` then `execute(parsed)`: the receiver is already
        # bound into the method object, so the call is spelled with no receiver at all
        # and the argument bound to `self` while the real parameter stayed clean.
        bound_alias = isinstance(node.func, ast.Name) and any(
            "." in spelling and self.instance_types.get(spelling.rpartition(".")[0])
            for spelling in self.callable_aliases.get(node.func.id) or ()
        )
        # A partial that pre-bound arguments shifts the rest left by that much, so the
        # helper's second parameter is the wrapper's first.
        partial_offset = (
            self.alias_offsets.get(node.func.id, 0) if isinstance(node.func, ast.Name) else 0
        )
        offset = (
            1
            if self.state.is_method.get(target)
            and (
                constructing
                or bound_alias
                or (
                    isinstance(node.func, ast.Attribute)
                    and not self._receiver_spelled_out(node.func, target)
                )
            )
            else 0
        ) + partial_offset
        for position, argument in enumerate(node.args):
            reason = self.tainted(argument)
            if not reason:
                continue
            if isinstance(argument, ast.Starred):
                # `execute(*json.loads(blob))` spreads over the parameters from here on,
                # and how many is not knowable, so all of them from this position are
                # bound. Treating it as one positional tainted only the first, and a
                # second element reaching subprocess.run was reported nowhere.
                for parameter in params[position + offset :]:
                    if parameter not in ("self", "cls"):
                        self._bind(bound, parameter, reason)
                continue
            index = position + offset
            # `def execute(*commands)`: positions at or past the vararg all land in it.
            # The flattened parameter list contains the vararg's name once, so anything
            # beyond that position was dropped and a value executed out of commands[1]
            # was reported nowhere.
            vararg = self.state.varargs.get(target)
            vararg_index = params.index(vararg) if vararg in params else None
            if vararg_index is not None and index >= vararg_index:
                self._bind(bound, vararg, reason)
            elif index < len(params):
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

    def _receiver_spelled_out(self, func: ast.Attribute, target) -> bool:
        """`Runner.execute(runner, ...)`: the receiver is already in `node.args`.

        Attribute syntax alone used to shift every argument one place right, so on an
        explicit unbound call the last argument was mapped past the end of the parameter
        list and dropped, and a sink inside the method reading it was reported nowhere.
        The receiver is written out exactly when the head names the class the method was
        found on, which is what tells `Runner.execute(...)` from `runner.execute(...)`.
        """
        # A classmethod is bound even on the `Runner.execute(...)` spelling, because
        # Python supplies `cls`. Treating that as an explicit receiver bound the first
        # real argument to `cls` and dropped the rest, which is the same loss the offset
        # fix was removing.
        if self.state.is_classmethod.get(target):
            return False
        head = _call_name(func.value)
        if not head or head in ("self", "cls") or head in self.instance_types:
            return False
        owner = target[1].rpartition(".")[0]
        return bool(owner) and head.rpartition(".")[2] == owner.rpartition(".")[2]

    def _check_sink(self, node: ast.Call) -> None:
        names = self.facts.canonicals(_call_name(node.func))
        matched = _matches_any(names, SINKS)
        candidates = [matched] if matched is not None else []
        if not candidates:
            # `loader = importlib.import_module` then `loader(name)`. Matched only under
            # the textual name `loader`, so an ordinary local alias walked past the gate.
            # getattr is not in SINKS; it has its own check because it needs the holder
            # inspected, so an alias of it has to go there rather than into this table.
            held = tuple(self.sink_aliases.get(_call_name(node.func)) or ())
            if isinstance(node.func, ast.Attribute):
                # Stored on an attribute by another method, so the local table knows
                # nothing about it.
                for key in self._attr_keys(node.func):
                    held = held + tuple(self.state.attr_sink_aliases.get(key) or ())
            candidates = [aliased for aliased in held if aliased not in _NOT_TABLE_SINKS]
        if not candidates:
            self._check_getattr(node)
            return
        offset = self.alias_offsets.get(_call_name(node.func), 0)
        for sink in candidates:
            self._check_one_sink(node, sink, offset)

    def _check_one_sink(
        self,
        node: ast.Call,
        sink: str,
        offset: int = 0,
    ) -> None:
        positions, keywords = SINKS[sink]
        # `offset` is what a `functools.partial` already bound, so the sink's position 1
        # is the wrapper's position 0. A watched position that the partial itself filled
        # goes negative and was already checked where it was bound.
        positions = tuple(index - offset for index in positions if index >= offset)
        for index in positions:
            if index < len(node.args):
                reason = self.tainted(node.args[index])
                if reason:
                    self._record(node, sink, reason, _short(node.args[index]))
                    return
            # A single `*values` can expand into any position from where it sits, so a
            # sink that watches position 1 saw one argument and looked no further:
            # `sys.path.insert(*parsed)` executed an attacker-controlled entry with
            # nothing reported. Same conservative reading the callee binding uses.
            for position, argument in enumerate(node.args):
                if position > index or not isinstance(argument, ast.Starred):
                    continue
                reason = self.tainted(argument)
                if reason:
                    self._record(node, sink, reason, _short(argument))
                    return
        for keyword in node.keywords:
            if keyword.arg is None:
                # `importlib.import_module(**json.loads(blob))`. The expansion can supply
                # the sink's own argument, and the earlier handling for this only covered
                # propagation into a first-party callee, so a direct sink skipped it.
                reason = self.tainted(keyword.value)
                if reason:
                    self._record(node, sink, reason, _short(keyword.value))
                    return
                continue
            if keyword.arg in keywords:
                reason = self.tainted(keyword.value)
                if reason:
                    self._record(node, sink, reason, _short(keyword.value))
                    return

    def _check_getattr(self, node: ast.Call) -> None:
        """`getattr(transformers, tainted)` resolves an arbitrary name in a namespace."""
        # Through canonicals, not the raw spelling: `from builtins import getattr as
        # resolve` left the holder-alias support in place while the alias of this sink
        # itself was skipped. A local alias counts too.
        called = _call_name(node.func)
        is_getattr = _matches_any(
            self.facts.canonicals(called), {"getattr", "builtins.getattr"}
        ) is not None or "getattr" in self._held_sinks(called, node)
        if not is_getattr or len(node.args) < 2:
            return
        holder = node.args[0]
        holder_names = (
            [candidate.split(".")[0] for candidate in self.facts.canonicals(_call_name(holder))]
            if not isinstance(holder, ast.Call)
            else []
        )
        # The spelling as written, not only the canonical target: `import x.y as ns`
        # canonicalises to `x.y`, whose head is `x`, so matching the alias table on the
        # canonical head never saw `ns` and the widening did nothing for an aliased import.
        written = _call_name(holder).split(".")[0] if not isinstance(holder, ast.Call) else ""
        is_module_ish = (
            any(held in MODULE_ISH_NAMES for held in holder_names)
            or (written in self.facts.module_aliases if written else False)
            or any(held in self.facts.module_aliases for held in holder_names)
        ) or (
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

    def _held_sinks(self, called: str, node: ast.Call) -> tuple:
        """Sinks the call target can be, from the local table and the shared one."""
        held = tuple(self.sink_aliases.get(called) or ())
        if isinstance(node.func, ast.Attribute):
            for key in self._attr_keys(node.func):
                held = held + tuple(self.state.attr_sink_aliases.get(key) or ())
        return held

    def _propagate_into_mapped(self, node: ast.Call) -> None:
        """`map(execute, parsed["commands"])` calls `execute` with every element.

        The result carried the taint onward, which is right, but the callback itself was
        never analysed with a tainted parameter, so a `subprocess.run` inside it ran
        every attacker-chosen command with nothing reported. The element is bound to the
        callback's first parameter, which is what `map` and `filter` do.
        """
        if (
            _matches_any(self.facts.canonicals(_call_name(node.func)), {"map", "filter"}) is None
            or len(node.args) < 2
        ):
            return
        # One positional slot per iterable, in order, which is how `map` calls the
        # callback: putting a tainted second iterable in slot 0 tainted the wrong
        # parameter and left the one that reaches the sink clean.
        if not any(self.tainted(iterable) for iterable in node.args[1:]):
            return
        invoked = ast.Call(func = node.args[0], args = list(node.args[1:]), keywords = [])
        ast.copy_location(invoked, node)
        self._propagate_into_callee(invoked)

    def _check_torch_load(self, node: ast.Call) -> None:
        """`torch.load(downloaded, weights_only = False)` unpickles attacker bytes.

        Narrow deliberately. The supported torch floor is 2.6, where `weights_only`
        defaults to True, so a call that does not mention it is already restricted to
        tensors and reporting it would flag most weights loads in this tree for nothing.
        What is unsafe is turning it off, or handing it a value that cannot be read as
        True here, and that is what this reports.
        """
        called = _call_name(node.func)
        if _matches_any(
            self.facts.canonicals(called), TORCH_LOAD_NAMES
        ) is None and TORCH_LOAD_ALIAS not in self._held_sinks(called, node):
            return
        weights_only = next(
            (keyword for keyword in node.keywords if keyword.arg == "weights_only"), None
        )
        if weights_only is None:
            return
        value = weights_only.value
        if isinstance(value, ast.Constant) and value.value is True:
            return
        if isinstance(value, ast.Name) and value.id in self.true_names:
            return
        if isinstance(value, ast.Name) and value.id in self.facts.module_true_names:
            return
        for argument in list(node.args[:1]) + [
            keyword.value for keyword in node.keywords if keyword.arg == "f"
        ]:
            reason = self.tainted(argument)
            if reason:
                self._record(node, TORCH_LOAD_SINK, reason, _short(argument))
                return

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
            # A local name bound to True counts: `enabled = True` then
            # `trust_remote_code = enabled` runs repository code just the same, and only
            # the literal at the call was recognised.
            written_true = (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is True
            ) or (isinstance(keyword.value, ast.Name) and keyword.value.id in self.true_names)
            if written_true:
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
        # `relative::Class.attr` -> the classes an instance held there was built from.
        # Shared rather than per-function, because `__init__` stores the object and a
        # different method calls through it, so a per-pass local map lost the binding.
        self.attr_instances: dict[str, tuple] = {}
        self.pending_attrs: dict[str, str] = {}
        # `relative::Class.attr` -> sinks stored on that attribute. Shared for the same
        # reason the held instances are: `self.loader = importlib.import_module` in the
        # constructor and `self.loader(parsed)` in another method are separate passes.
        self.attr_sink_aliases: dict[str, tuple] = {}
        self.tainted_globals: dict[str, str] = {}
        self.returns_tainted: dict[tuple[Path, str], str] = {}
        self.params: dict[tuple[Path, str], list[str]] = {}
        self.is_method: dict[tuple[Path, str], bool] = {}
        # (file, qualname) -> decorated @classmethod. Python supplies `cls` even on the
        # `Runner.execute(...)` spelling, so the receiver offset has to stay there.
        self.is_classmethod: dict[tuple[Path, str], bool] = {}
        # (file, qualname) -> the name of the callee's **kwargs parameter, if it has one
        self.star_kwargs: dict[tuple[Path, str], str] = {}
        # (file, qualname) -> the name of the callee's *args parameter, if it has one
        self.varargs: dict[tuple[Path, str], str] = {}

    def snapshot(self) -> str:
        return json.dumps(
            {
                "params": sorted(
                    f"{path}::{qualname}::{name}={reason}"
                    for (path, qualname), names in self.tainted_params.items()
                    for name, reason in names.items()
                ),
                "attrs": sorted(f"{key}={reason}" for key, reason in self.tainted_attrs.items()),
                "held": sorted(
                    f"{key}={','.join(types)}" for key, types in self.attr_instances.items()
                ),
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

    # Names known to hold True, per owner scope, chained to a fixpoint so `enabled =
    # True; remote = enabled` counts. Only a literal True was accepted in the spellings
    # below, so `kwargs = {"trust_remote_code": enabled}` splatted into a loader enabled
    # repository code with nothing reported, while the same name passed as a keyword
    # was already caught. Flow-insensitive, like the keyword check it mirrors.
    true_by_owner: dict[str, set] = {}
    assignments: dict[str, list] = {}
    for node in ast.walk(facts.tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.Name)):
            owner = owners.get(id(node), "<module>")
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(owner, []).append((target.id, node.value))
    for owner, pairs in assignments.items():
        known = set(facts.module_true_names)
        changed = True
        while changed:
            changed = False
            for name, value in pairs:
                if name in known:
                    continue
                if (isinstance(value, ast.Constant) and value.value is True) or (
                    isinstance(value, ast.Name) and value.id in known
                ):
                    known.add(name)
                    changed = True
        true_by_owner[owner] = known

    def is_true(value: ast.AST, node: ast.AST) -> bool:
        if isinstance(value, ast.Constant):
            return value.value is True
        if isinstance(value, ast.Name):
            owner = owners.get(id(node), "<module>")
            return value.id in true_by_owner.get(owner, facts.module_true_names)
        return False

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
                    and is_true(value, node)
                ):
                    record(node, "dict", _short(node))
        elif isinstance(node, ast.Call) and _call_name(node.func).rpartition(".")[2] == "dict":
            # `kwargs = dict(trust_remote_code = True)` is the same thing as the literal
            # and splats into a loader the same way, but it is a Call rather than a Dict,
            # so neither this scan nor the call-site keyword check saw it.
            for keyword in node.keywords:
                if keyword.arg == "trust_remote_code" and is_true(keyword.value, node):
                    record(node, "dict call", _short(node))
        elif isinstance(node, ast.Assign):
            if not is_true(node.value, node):
                continue
            for target in node.targets:
                # The bare-name spelling stays literal-only: `trust_remote_code = enabled`
                # is how a parameter gets forwarded, and the keyword check already reads
                # the name at the call site where it is actually used.
                if (
                    isinstance(target, ast.Name)
                    and target.id == "trust_remote_code"
                    and isinstance(node.value, ast.Constant)
                ):
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


def _queue_class_bodies(node: ast.ClassDef, prefix: str, queue: list) -> None:
    """`(qualname, statements)` for this class body and every class nested in it.

    The methods are left out: they are indexed and analysed under their own qualnames.
    """
    name = f"{prefix}.{node.name}" if prefix else node.name
    queue.append(
        (
            f"<class {name}>",
            [
                inner
                for inner in ast.iter_child_nodes(node)
                if not isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ],
        )
    )
    for inner in ast.iter_child_nodes(node):
        if isinstance(inner, ast.ClassDef):
            _queue_class_bodies(inner, name, queue)


_CLASS_BODIES: dict = {}


def _class_bodies(facts: _FileFacts) -> list:
    """Every class body in the file, collected once and shared.

    Both the fixpoint and the reporting pass need these, and collecting them twice is
    how the two halves of a check drift apart, so there is one list and one memo.
    """
    cached = _CLASS_BODIES.get(facts.path)
    if cached is not None:
        return cached
    bodies: list = []
    for child in ast.iter_child_nodes(facts.tree):
        if isinstance(child, ast.ClassDef):
            _queue_class_bodies(child, "", bodies)
    _CLASS_BODIES[facts.path] = bodies
    return bodies


def _class_body_owner(qualname: str) -> str:
    """`<class Config>` -> `Config`, the name an attribute key is built from."""
    if qualname.startswith("<class ") and qualname.endswith(">"):
        return qualname[len("<class ") : -1]
    return qualname


def _publish_class_attributes(facts: _FileFacts, qualname: str, visitor, state) -> None:
    """A name bound in a class body IS a class attribute, so the binding has to leave.

    `class Config: module = json.load(...)["module"]` kept the taint inside the body's
    own pass and discarded it, so both `Config.module` and a method's `self.module`
    read clean everywhere else. Published under the key attribute reads already use,
    with the same tier-A-wins rule as every other write.
    """
    owner = _class_body_owner(qualname)
    for name, reason in visitor.local_reasons.items():
        key = f"{facts.relative}::{owner}.{name}"
        if state.pending_attrs.get(key, NAMED_PARAM_REASON) == NAMED_PARAM_REASON:
            state.pending_attrs[key] = reason
    # The types too, not only the taint: `class App: runner = Runner()` is how a shared
    # collaborator is declared, and discarding it meant `self.runner.execute(parsed)`
    # resolved to nothing and a sink inside that method was reported nowhere.
    for name, constructed in visitor.instance_types.items():
        key = f"{facts.relative}::{owner}.{name}"
        for candidate in constructed:
            state.attr_instances[key] = _with(state.attr_instances.get(key), candidate)
    # And the sinks: `class Hooks: loader = importlib.import_module` is a sink stored on
    # a class attribute, and `Hooks.loader(parsed)` could not recover it because only
    # the taint reasons left the body.
    for name, sinks in visitor.sink_aliases.items():
        if name in facts.module_sink_aliases and sinks == facts.module_sink_aliases[name]:
            continue
        key = f"{facts.relative}::{owner}.{name}"
        for candidate in sinks:
            state.attr_sink_aliases[key] = _with(state.attr_sink_aliases.get(key), candidate)


# Statement kinds that can bind a class attribute. A body of nothing but a docstring,
# `pass`, bare annotations and imports publishes nothing, and skipping those in the
# fixpoint keeps the added work off the thousands of classes in these trees.
_BINDING_STATEMENTS = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.With, ast.If, ast.Try)


def _unpinned_code_fetches(facts: _FileFacts, reached: frozenset) -> list[dict]:
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
    has already decided: `reached` carries the identity of each download whose value
    arrived at an import sink. Correlation, not proximity, and per download rather than
    per function: a function that pins the code it imports and separately downloads
    weights used to have the weights fetch reported, because a function-wide boolean
    cannot say which of the two the sink actually consumed.

    The identity is the fetch's own file and line, so the set is global rather than
    keyed by the sink's qualname. Keying it by the sink meant a download returned by one
    helper and imported by another was never correlated at all: dropping a `revision`
    changed only the helper, the sink's allowance stayed valid, and nothing was reported.
    """
    findings: list[dict] = []
    for qualname, node in sorted(facts.functions.items()):
        if not reached:
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
            # `revision = None` reaches the library as the same unpinned default as an
            # omitted keyword, so a presence-only test let the explicit spelling through.
            if any(
                keyword.arg == "revision"
                and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
                for keyword in child.keywords
            ):
                continue
            # Same identity the taint reason carries, so this is the download the sink
            # read and not merely one of the downloads in the same body.
            if f"{source}()@{facts.relative}:{child.lineno}" not in reached:
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


# Registered here rather than beside each table so the list is in one place and a new
# table is cached only once someone has thought about its lifetime.
_CACHEABLE_TABLES.update(
    id(table)
    for table in (
        SINKS,
        SINKS_GATED_ELSEWHERE,
        UNTRUSTED_CALLS,
        UNTRUSTED_METHODS,
        UNTRUSTED_PARAM_NAMES,
        MODULE_ISH_NAMES,
    )
)

# How many package layers a re-export is followed through. Three covers `pkg` ->
# `pkg.api` -> `pkg.api.impl`, and the bound is what makes a circular re-export
# terminate rather than spin.
_REEXPORT_HOPS = 8
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
    # Carried with the reasons, because a method called before the line that constructs
    # the object would otherwise never resolve on any pass.
    instances: dict[str, str] = {}
    aliases: dict[str, str] = {}
    # Carried like the others: a source bound on a loop backedge, after its first
    # lexical use, was dropped between passes and the sink was never revisited with the
    # name recognised as a deserialiser.
    sources: dict[str, str] = {}
    callables: dict[str, str] = {}
    # A flag that becomes true on a loop backedge: every pass visited the loader call
    # before rediscovering the assignment, so the second runtime iteration enabled
    # remote code and nothing was reported.
    trues: set = set()
    # Carried like the aliases themselves: a partial built on a loop backedge otherwise
    # had its layout discarded between passes and the shifted call read clean again.
    offsets: dict[str, int] = {}
    visitor = None
    for _ in range(_LOCAL_BOUND):
        visitor = _TaintPass(facts, qualname, state)
        visitor.local_reasons.update(reasons)
        visitor.instance_types.update(instances)
        visitor.sink_aliases.update(aliases)
        visitor.source_aliases.update(sources)
        visitor.callable_aliases.update(callables)
        visitor.true_names.update(trues)
        visitor.alias_offsets.update(offsets)
        visitor.seed_defaults()
        for child in nodes:
            visitor.visit(child)
        if qualname in facts.lambdas:
            # A lambda has no `return` statement; its body is the value it hands back,
            # so without this the wrapper had no tainted summary and its callers read
            # clean even once the parameter was bound.
            body = facts.functions[qualname].body
            visitor._note_output(body)
        if (
            visitor.local_reasons == reasons
            and visitor.instance_types == instances
            and visitor.sink_aliases == aliases
            and visitor.source_aliases == sources
            and visitor.callable_aliases == callables
            and visitor.true_names == trues
            and visitor.alias_offsets == offsets
        ):
            return visitor, True
        reasons = dict(visitor.local_reasons)
        instances = dict(visitor.instance_types)
        aliases = dict(visitor.sink_aliases)
        sources = dict(visitor.source_aliases)
        callables = dict(visitor.callable_aliases)
        trues = set(visitor.true_names)
        offsets = dict(visitor.alias_offsets)
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
    _CALL_NAMES.clear()
    _CLASS_BODIES.clear()
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
    # Published on the index so a package re-export can be followed to the file that
    # defines the helper. After the loop, because every file has to be parsed first.
    index.facts = facts_by_path

    state = _State()
    for path, facts in sorted(facts_by_path.items()):
        for qualname, node in sorted(facts.functions.items()):
            key = (path, qualname)
            state.params[key] = facts.params[qualname]
            # Class containment, not the receiver's spelling. `def execute(this,
            # command)` is a valid method and read as a plain function, so a tainted
            # argument bound to `this` and the parameter the sink consumed stayed clean.
            state.is_method[key] = qualname in facts.methods
            if any(
                _call_name(decorator).rpartition(".")[2] == "classmethod"
                for decorator in getattr(node, "decorator_list", ())
            ):
                state.is_classmethod[key] = True
            star = facts.star_kwargs.get(qualname)
            if star:
                state.star_kwargs[key] = star
            vararg = facts.varargs.get(qualname)
            if vararg:
                state.varargs[key] = vararg
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
            # Class bodies too, for the attributes they bind. Only functions were
            # settled here, so a class body's write was discovered in the reporting
            # pass, after the last thing that could have read it had already run.
            for class_qualname, statements in _class_bodies(facts):
                if not any(isinstance(n, _BINDING_STATEMENTS) for n in statements):
                    continue
                class_visitor, _ = _settle(facts, class_qualname, statements, state)
                if class_visitor is not None:
                    _publish_class_attributes(facts, class_qualname, class_visitor, state)
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
    # (facts, the download identities that reached an import in that file). The rule
    # runs after the loop against the union, so a fetch in one file and the import it
    # feeds in another are correlated.
    pending_fetches: list = []
    all_reached: set = set()
    for path, facts in sorted(facts_by_path.items()):
        reached: set = set()
        for qualname, node in sorted(facts.functions.items()):
            visitor, converged = _settle(facts, qualname, list(ast.iter_child_nodes(node)), state)
            if not converged:
                unconverged.add(f"{facts.relative}::{qualname}")
            found = visitor.findings if visitor is not None else []
            reached.update(
                f["why"]
                for f in found
                if f["sink"] in _IMPORT_SINKS
                and any(f["why"].startswith(prefix) for prefix in _DOWNLOADS)
            )
            findings.extend(found)
        body = []
        class_bodies = _class_bodies(facts)
        for child in ast.iter_child_nodes(facts.tree):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(child, ast.ClassDef):
                # A class body executes at import time, and it was skipped outright, so a
                # sink sitting in one was never scanned at all. Only the statements the
                # body itself runs: the methods are analysed under their own qualnames.
                # Each body is settled on its own, because flattening them into the module
                # merged one class's attributes with another's and with the real globals,
                # so a safe `command` in one class failed on a parsed one in a different
                # class entirely.
                # Recursively, because Python runs a nested class body while defining
                # its parent and the nested node was filtered out of the only worklist,
                # so an import-time sink inside `class Outer: class Inner:` was never
                # visited at all. Each body keeps its own namespace. Collected by
                # `_class_bodies`, which the fixpoint reads from as well.
                continue
            body.append(child)
        # A context digest for the module body and each class body too. Only indexed
        # functions had one, so a finding at module or class scope was baselined with an
        # empty digest, and weakening the validation or consent guard around a reviewed
        # sink kept its allowance as long as the call text itself did not change, which
        # is the one thing the digest exists to stop.
        facts.contexts.setdefault("<module>", _norm_body_hash(body))
        for qualname, statements in class_bodies:
            facts.contexts.setdefault(qualname, _norm_body_hash(statements))
        for qualname, statements in class_bodies:
            class_visitor, class_converged = _settle(facts, qualname, statements, state)
            if not class_converged:
                unconverged.add(f"{facts.relative}::{qualname}")
            if class_visitor is not None:
                findings.extend(class_visitor.findings)
        module_visitor, module_converged = _settle(facts, "<module>", body, state)
        if not module_converged:
            unconverged.add(f"{facts.relative}::<module>")
        if module_visitor is not None:
            findings.extend(module_visitor.findings)
        findings.extend(_remote_code_defaults(facts))
        # Deferred until every file has been settled, because a download returned by a
        # helper in one file and imported in another is only correlated once both sides
        # have been seen.
        pending_fetches.append(facts)
        all_reached.update(reached)

    frozen = frozenset(all_reached)
    for facts in pending_fetches:
        findings.extend(_unpinned_code_fetches(facts, frozen))

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
