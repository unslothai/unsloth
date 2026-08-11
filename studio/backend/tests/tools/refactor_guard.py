# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic guard for refactors of the tool-call parsing / stripping stack.

Three independent checks, none of which trusts a reading of the diff:

1. **AST inventory** - every top-level name, its signature, decorators, and every
   ``re.compile`` pattern literal, for each guarded module. A refactor that drops or
   renames a symbol other modules import shows up as an inventory diff.
2. **Golden outputs** - every guarded function is driven over a corpus of tool-call
   text and its output recorded. A refactor must reproduce those bytes exactly.
   Each function is also asserted idempotent where that is meaningful
   (``f(f(x)) == f(x)`` for the strip functions).
3. **Patch-target routing** - the test suite patches module globals by string
   (``patch("core.inference.llama_cpp.subprocess.run")``). Moving a function to
   another module leaves those patches pointing at a namespace nobody reads, so the
   test still passes while exercising unpatched code. This check asserts every
   string patch target still resolves.

Usage::

    python tests/tools/refactor_guard.py snapshot   # record the baseline
    python tests/tools/refactor_guard.py verify     # compare against it
    python tests/tools/refactor_guard.py twins      # report healing/parser divergence

``test_refactor_guard.py`` runs ``verify`` in CI.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import random
import re
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

BASELINE_DIR = BACKEND_ROOT / "tests" / "data" / "refactor_guard"

# Modules whose top-level surface is pinned. Import path -> file path.
GUARDED_MODULES = {
    "core.tool_healing": BACKEND_ROOT / "core" / "tool_healing.py",
    "core.inference.tool_call_parser": BACKEND_ROOT / "core" / "inference" / "tool_call_parser.py",
    "core.inference.llama_cpp": BACKEND_ROOT / "core" / "inference" / "llama_cpp.py",
    "core.inference.inference": BACKEND_ROOT / "core" / "inference" / "inference.py",
    "core.inference.safetensors_agentic": (
        BACKEND_ROOT / "core" / "inference" / "safetensors_agentic.py"
    ),
}

# Modules whose behaviour is pinned by golden outputs. Importing these must not pull
# in the inference stack, so llama_cpp is deliberately absent: its strip closure is
# exercised through the streaming replay in test_refactor_guard.py instead.
BEHAVIOUR_MODULES = ("core.tool_healing", "core.inference.tool_call_parser")


# ─────────────────────────── 1. AST inventory ───────────────────────────


def _pattern_literals(tree: ast.Module) -> dict:
    """Every ``re.compile(...)`` call in the module, as a sorted multiset of its source.

    A regex rewrite is the easiest way to silently change stripping behaviour, so the
    literals are pinned separately from the symbol table. Keyed by the call text rather
    than by line number, so moving or deleting unrelated code does not register as a
    pattern change.
    """
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compile"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re"
        ):
            calls.append(ast.unparse(node))
    counts = {}
    for call in calls:
        counts[call] = counts.get(call, 0) + 1
    return counts


def ast_inventory() -> dict:
    inventory = {}
    for mod_name, path in GUARDED_MODULES.items():
        tree = ast.parse(path.read_text())
        symbols = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols[node.name] = {
                    "kind": "async def" if isinstance(node, ast.AsyncFunctionDef) else "def",
                    "signature": ast.unparse(node.args),
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                }
            elif isinstance(node, ast.ClassDef):
                methods = {}
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods[sub.name] = {
                            "signature": ast.unparse(sub.args),
                            "decorators": [ast.unparse(d) for d in sub.decorator_list],
                        }
                symbols[node.name] = {
                    "kind": "class",
                    "bases": [ast.unparse(b) for b in node.bases],
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "methods": methods,
                }
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbols[target.id] = {"kind": "assign"}
        inventory[mod_name] = {"symbols": symbols, "regexes": _pattern_literals(tree)}
    return inventory


def runtime_inventory() -> dict:
    """``dir()`` plus signatures, which catches re-export aliasing the AST cannot see."""
    out = {}
    for mod_name in BEHAVIOUR_MODULES:
        module = importlib.import_module(mod_name)
        entry = {"dir": sorted(n for n in dir(module) if not n.startswith("__"))}
        signatures = {}
        for name in entry["dir"]:
            obj = getattr(module, name)
            if inspect.isfunction(obj):
                signatures[name] = f"{obj.__module__}.{obj.__qualname__}{inspect.signature(obj)}"
        entry["functions"] = signatures
        out[mod_name] = entry
    return out


# ─────────────────────────── 2. Corpus + golden outputs ───────────────────────────

# Literal fragments spliced by the fuzzer. Every serialization the parsers claim to
# handle appears here, plus the shapes that historically broke them: markup nested in
# an argument value, a rehearsal inside a fenced code block, truncated tails.
_FRAGMENTS = (
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>',
    '<tool_call>{"name": "search", "arguments": {"q": "</tool_call> literal"}}</tool_call>',
    '<tool_call>{"name": "trunc", "arguments": {"a": ',
    "<function=get_weather><parameter=city>Paris</parameter></function>",
    '<function name="get_weather"><parameter=city>Paris</parameter></function>',
    "<function=broken><parameter=city>Paris",
    '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>',
    "<|tool_call>call:get_weather{city:Paris",
    'call:get_weather{city:<|"|>Paris<|"|>}',
    '[TOOL_CALLS]get_weather{"city": "Paris"}',
    '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]',
    '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}',
    'get_weather[ARGS]{"city": "Paris"}',
    "get_weather[ARGS]{",
    "get_weather[ARGS]",
    'unlisted_tool[ARGS]{"city": "Paris"}',
    '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}',
    '<|python_tag|>get_weather.call(city="Paris")',
    '{"name": "get_weather", "parameters": {"city": "Paris"}}',
    '<|message_model|>get_weather<|content_invoke_tool_json|>'
    '{"name": "get_weather", "args": {"city": "Paris"}}<|end_message|>',
    "<think>I should call get_weather[ARGS]{} but not really</think>",
    "<think>unclosed reasoning",
    "```\nget_weather[ARGS]{\"city\": \"Paris\"}\n```",
    "`get_weather[ARGS]{}`",
    "Sure, let me look that up for you.",
    "The weather in Paris is 18 degrees.",
    "",
    "{",
    "}",
    "[",
    "]",
    '[{"a": 1}]',
    "[}]",
    '["a}b"]',
    "\\",
    '{"a": "\\""}',
)

_ENABLED_NAMES = ("get_weather", "search", "trunc", "broken")


def build_corpus(seed: int = 20260811, count: int = 600) -> list:
    """Deterministic corpus: every fragment alone, every ordered pair, then random splices."""
    corpus = list(_FRAGMENTS)
    for left in _FRAGMENTS:
        for right in _FRAGMENTS:
            corpus.append(left + right)
    rng = random.Random(seed)
    joiners = ("", " ", "\n", "\n\n", " and then ", "\n```\n")
    for _ in range(count):
        parts = [rng.choice(_FRAGMENTS) for _ in range(rng.randint(2, 5))]
        corpus.append(rng.choice(joiners).join(parts))
    # Dedupe while keeping order so the golden file is stable across runs.
    return list(dict.fromkeys(corpus))


def _drive(func, text: str):
    """Call ``func`` with ``text`` however its signature wants it.

    Returns a JSON-safe result, or a marker string when the function raises. An
    exception is itself pinned behaviour: a refactor that stops raising, or starts,
    is a change.
    """
    params = inspect.signature(func).parameters
    names = list(params)
    kwargs = {}
    if "enabled_tool_names" in params:
        kwargs["enabled_tool_names"] = set(_ENABLED_NAMES)
    if "final" in params and params["final"].default is not inspect.Parameter.empty:
        kwargs["final"] = True

    args = [text]
    # Positional index arguments (``brace_start``, ``start``, ``pos``, ``brace_pos``)
    # are driven from the first plausible offset in the text rather than 0, so the
    # balanced scanners are exercised on a real opening delimiter.
    for extra in names[1:]:
        if extra in kwargs or params[extra].kind == inspect.Parameter.KEYWORD_ONLY:
            continue
        if params[extra].default is not inspect.Parameter.empty:
            continue
        if extra in ("brace_start", "brace_pos"):
            args.append(max(text.find("{"), 0))
        elif extra == "start":
            args.append(max(text.find("["), 0))
        elif extra == "pos":
            args.append(len(text))
        else:
            return "<undrivable>"
    try:
        result = func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the exception type is the pinned value
        return f"<raised {type(exc).__name__}>"
    return _jsonable(result)


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if inspect.isgenerator(value):
        # A generator's repr carries its address, which is not reproducible. The
        # yielded values are the behaviour worth pinning.
        try:
            return [_jsonable(item) for item in value]
        except Exception as exc:  # noqa: BLE001 - raising mid-iteration is pinned too
            return f"<raised {type(exc).__name__}>"
    return repr(value)


def _guarded_functions(mod_name):
    """Single-text-argument functions in a behaviour module, in a stable order."""
    module = importlib.import_module(mod_name)
    out = []
    for name in sorted(dir(module)):
        if name.startswith("__"):
            continue
        obj = getattr(module, name)
        if not inspect.isfunction(obj) or obj.__module__ != mod_name:
            continue
        params = list(inspect.signature(obj).parameters.values())
        if not params:
            continue
        if params[0].annotation not in (str, "str"):
            continue
        out.append((name, obj))
    return out


def golden_outputs(corpus) -> dict:
    """A digest per guarded function over the whole corpus.

    Storing every output would be a 7 MB file in the repo for no extra guarantee: any
    byte that changes anywhere in a function's results changes its digest. The corpus is
    rebuilt deterministically from ``build_corpus`` rather than checked in, so its own
    digest is recorded too and a corpus edit reports as exactly that instead of as 40
    unrelated behaviour changes.
    """
    out = {"corpus": _digest([len(corpus), corpus])}
    for mod_name in BEHAVIOUR_MODULES:
        per_module = {}
        for name, func in _guarded_functions(mod_name):
            per_module[name] = _digest([_drive(func, text) for text in corpus])
        out[mod_name] = per_module
    return out


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=repr)
    return hashlib.sha256(payload.encode()).hexdigest()


def first_divergence(corpus, mod_name, func_name):
    """Recompute one function's outputs so a digest mismatch can be localised.

    Only useful next to a checkout of the previous revision, which is what you want when
    a digest moves: run this on both sides and compare.
    """
    module = importlib.import_module(mod_name)
    func = getattr(module, func_name)
    return [{"input": text, "output": _drive(func, text)} for text in corpus]


def idempotence_failures(corpus) -> list:
    """``f(f(x)) != f(x)`` for the str -> str functions.

    Stripping twice must not remove more than stripping once; a stripper that is not
    idempotent produces different display text depending on how a stream is chunked.
    """
    failures = []
    for mod_name in BEHAVIOUR_MODULES:
        for name, func in _guarded_functions(mod_name):
            # Only the strip family is meant to be a text -> text projection. Feeding a
            # parser's output (a tool name, a JSON blob) back into it proves nothing.
            if "strip" not in name:
                continue
            for text in corpus:
                once = _drive(func, text)
                if not isinstance(once, str) or once.startswith("<raised "):
                    continue
                twice = _drive(func, once)
                if twice != once:
                    failures.append(
                        {
                            "module": mod_name,
                            "function": name,
                            "input": text,
                            "once": once,
                            "twice": twice,
                        }
                    )
                    break  # one witness per function is enough to fail the check
    return failures


# ─────────────────────────── 3. Patch-target routing ───────────────────────────

_PATCH_TARGET_RE = re.compile(
    r"""(?:mock\.)?(?:patch|monkeypatch\.setattr)\(\s*["']((?:core|routes|utils|state)\.[\w.]+)["']"""
)


def patch_targets(tests_dir=None) -> dict:
    """String patch targets found in the test suite, grouped by module."""
    tests_dir = tests_dir or (BACKEND_ROOT / "tests")
    targets = {}
    for path in sorted(tests_dir.rglob("test_*.py")):
        for match in _PATCH_TARGET_RE.finditer(path.read_text(errors="ignore")):
            dotted = match.group(1)
            targets.setdefault(dotted, []).append(str(path.relative_to(BACKEND_ROOT)))
    return targets


def unresolvable_patch_targets(targets=None) -> list:
    """Targets that no longer resolve to an attribute of an importable module.

    Splits ``a.b.c`` at every dot: the longest importable prefix is the module, the
    remainder must be reachable by ``getattr``.
    """
    targets = targets if targets is not None else patch_targets()
    broken = []
    for dotted, users in sorted(targets.items()):
        parts = dotted.split(".")
        obj = None
        for split in range(len(parts) - 1, 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:split]))
            except Exception:  # noqa: BLE001 - an unimportable prefix is just not the module
                continue
            rest = parts[split:]
            break
        else:
            broken.append({"target": dotted, "reason": "no importable module prefix", "tests": users})
            continue
        for attr in rest:
            # ``core.inference`` resolves attributes through a PEP 562 ``__getattr__``
            # that imports a submodule, so a missing optional dependency surfaces here
            # as an ImportError rather than an AttributeError. That is an environment
            # gap, not a broken target: record it separately so a genuinely missing
            # attribute is never hidden behind it.
            try:
                found = hasattr(obj, attr)
            except Exception as exc:  # noqa: BLE001
                broken.append(
                    {
                        "target": dotted,
                        "reason": f"unresolvable in this environment: {type(exc).__name__}",
                        "tests": users,
                        "environment": True,
                    }
                )
                break
            if not found:
                # A missing name on a package is ambiguous: the symbol may genuinely be
                # gone, or the submodule that defines it may just be unimportable here
                # (an optional dependency). Importing it directly tells the two apart, so
                # a real break is never written off as an environment gap.
                reason = f"missing attribute {attr!r}"
                environment = False
                if inspect.ismodule(obj):
                    try:
                        importlib.import_module(f"{obj.__name__}.{attr}")
                    except ImportError as exc:
                        reason = f"submodule {attr!r} is unimportable here: {exc}"
                        environment = True
                    except Exception:
                        pass
                    else:
                        found = True
                if not found:
                    entry = {"target": dotted, "reason": reason, "tests": users}
                    if environment:
                        entry["environment"] = True
                    broken.append(entry)
                    break
            obj = getattr(obj, attr, None)
    return broken


# ─────────────────────────── twins ───────────────────────────

# Names defined in both modules. Unifying them is the point of the refactor; this
# reports where they actually disagree on real input, so the choice of which body
# survives is made against evidence rather than against the diff.
TWIN_NAMES = (
    "_balanced_brace_end",
    "_balanced_bracket_end",
    "_inside_open_parameter",
    "_trim_param_value",
    "parse_tool_calls_from_text",
)


def twin_divergence(corpus) -> dict:
    healing = importlib.import_module("core.tool_healing")
    parser = importlib.import_module("core.inference.tool_call_parser")
    report = {}
    for name in TWIN_NAMES:
        h_func, p_func = getattr(healing, name, None), getattr(parser, name, None)
        if h_func is None or p_func is None:
            report[name] = {"status": "unified" if h_func is p_func else "missing"}
            continue
        if h_func is p_func:
            report[name] = {"status": "unified"}
            continue
        examples = []
        for text in corpus:
            h_out, p_out = _drive(h_func, text), _drive(p_func, text)
            # -1 and None are the two sentinels for "no match"; treat them as equal so
            # the report shows real disagreement rather than the known convention split.
            if (h_out in (-1, None)) and (p_out in (-1, None)):
                continue
            if h_out != p_out:
                examples.append({"input": text, "healing": h_out, "parser": p_out})
        report[name] = {
            "status": "diverges" if examples else "equivalent",
            "count": len(examples),
            "examples": examples[:5],
        }
    return report


# ─────────────────────────── CLI ───────────────────────────


def _write(name, payload):
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def _read(name):
    return json.loads((BASELINE_DIR / name).read_text())


def _diff(label, old, new):
    """Report the first differing JSON path, which is enough to locate the change."""
    if old == new:
        return []
    problems = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old:
                problems.append(f"{label}.{key}: added")
            elif key not in new:
                problems.append(f"{label}.{key}: REMOVED")
            else:
                problems.extend(_diff(f"{label}.{key}", old[key], new[key]))
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for index, (a, b) in enumerate(zip(old, new)):
            problems.extend(_diff(f"{label}[{index}]", a, b))
    elif (
        isinstance(old, list)
        and isinstance(new, list)
        and all(isinstance(v, str) for v in old + new)
    ):
        # A name list (``dir()``): report what joined or left, not both full lists.
        for name in sorted(set(new) - set(old)):
            problems.append(f"{label}: added {name!r}")
        for name in sorted(set(old) - set(new)):
            problems.append(f"{label}: REMOVED {name!r}")
    else:
        problems.append(f"{label}: {old!r} -> {new!r}")
    return problems


def snapshot():
    corpus = build_corpus()
    _write("ast_inventory.json", ast_inventory())
    _write("runtime_inventory.json", runtime_inventory())
    _write("golden_outputs.json", golden_outputs(corpus))
    _write("patch_targets.json", patch_targets())
    failures = idempotence_failures(corpus)
    _write("idempotence_baseline.json", failures)
    print(f"baseline written to {BASELINE_DIR}")
    print(f"  corpus: {len(corpus)} inputs")
    print(f"  patch targets: {len(patch_targets())}")
    print(f"  pre-existing idempotence failures: {len(failures)}")
    for failure in failures:
        print(f"    {failure['module']}.{failure['function']}")


def verify() -> int:
    corpus = build_corpus()
    problems = []
    problems += _diff("ast", _read("ast_inventory.json"), ast_inventory())
    problems += _diff("runtime", _read("runtime_inventory.json"), runtime_inventory())
    problems += _diff("golden", _read("golden_outputs.json"), golden_outputs(corpus))

    baseline_idempotence = {(f["module"], f["function"]) for f in _read("idempotence_baseline.json")}
    for failure in idempotence_failures(corpus):
        if (failure["module"], failure["function"]) not in baseline_idempotence:
            problems.append(
                f"idempotence.{failure['module']}.{failure['function']}: "
                f"f(f(x)) != f(x) for {failure['input']!r}"
            )

    for broken in unresolvable_patch_targets():
        problems.append(f"patch-target {broken['target']}: {broken['reason']} ({broken['tests'][0]})")

    if problems:
        print(f"FAIL: {len(problems)} difference(s)")
        for problem in problems[:60]:
            print(f"  {problem}")
        if len(problems) > 60:
            print(f"  ... and {len(problems) - 60} more")
        return 1
    print("OK: inventory, golden outputs, idempotence and patch targets all match")
    return 0


def twins():
    report = twin_divergence(build_corpus())
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if command == "snapshot":
        snapshot()
        return 0
    if command == "verify":
        return verify()
    if command == "twins":
        twins()
        return 0
    print(f"unknown command {command!r}; expected snapshot, verify or twins")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
