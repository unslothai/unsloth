# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic guard for refactors of the tool-call parsing / stripping stack.

Three independent checks, none of which trusts a reading of the diff:

1. **AST inventory** - every top-level name, signature, decorator and ``re.compile``
   literal per guarded module, so a dropped or renamed symbol shows up as a diff.
2. **Golden outputs** - every guarded function driven over a corpus of tool-call text,
   its output recorded, and the strip functions asserted idempotent.
3. **Patch-target routing** - the test suite patches module globals by string
   (``patch("core.inference.llama_cpp.subprocess.run")``). A moved function leaves those
   pointing at a namespace nobody reads, so the test passes while exercising unpatched
   code; this asserts every target still resolves.

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
from collections import Counter
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

# Modules whose behaviour is pinned by golden outputs. Importing these must not pull in
# the inference stack, so llama_cpp is deliberately absent.
BEHAVIOUR_MODULES = ("core.tool_healing", "core.inference.tool_call_parser")


# ─────────────────────────── 1. AST inventory ───────────────────────────


# Every ``re`` entry point carrying a pattern, not just ``compile``: an inline
# ``re.match(r"...")`` is as load-bearing as a compiled constant.
_RE_CALLS = frozenset(
    {"compile", "match", "search", "fullmatch", "sub", "subn", "split", "findall", "finditer"}
)


def _compiled_patterns(mod_name: str) -> dict:
    """Every compiled pattern reachable in the module's namespace, by its actual text.

    The AST half records call *source*, so a pattern interpolating a constant pins only
    that constant's name. ``.pattern`` off the live object pins what was compiled.
    """
    module = importlib.import_module(mod_name)
    out = {}
    for name in dir(module):
        obj = getattr(module, name, None)
        for index, item in enumerate(obj if isinstance(obj, (list, tuple)) else [obj]):
            if isinstance(item, re.Pattern):
                key = name if not isinstance(obj, (list, tuple)) else f"{name}[{index}]"
                out[key] = f"{item.pattern!r} flags={item.flags}"
    return out


def _pattern_literals(tree: ast.Module) -> dict:
    """Every ``re.compile(...)`` call in the module, as a multiset of its source.

    A regex rewrite is the easiest silent change to stripping behaviour, so the literals
    are pinned separately. Keyed by call text, so moving unrelated code is not a change.
    """
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _RE_CALLS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re"
        ):
            calls.append(ast.unparse(node))
    counts = {}
    for call in calls:
        counts[call] = counts.get(call, 0) + 1
    return counts


def ast_inventory() -> dict:
    """Top-level surface per guarded module, at the detail that module needs.

    The two restructured modules are pinned in full: signatures, decorators, methods and
    every ``re.compile`` literal. The rest are pinned by name and kind only, which is all
    "did this drop a symbol" needs; pinning their signatures too would fail on every
    later unrelated change to files as busy as ``llama_cpp.py``, and a baseline that gets
    regenerated reflexively guards nothing.
    """
    inventory = {}
    for mod_name, path in GUARDED_MODULES.items():
        detailed = mod_name in BEHAVIOUR_MODULES
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        symbols = {}
        for node in tree.body:
            if not detailed:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols[node.name] = {
                        "kind": "class" if isinstance(node, ast.ClassDef) else "def"
                    }
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            symbols[target.id] = {"kind": "assign"}
                # An annotated global is an AnnAssign, not an Assign, so it needs its own arm.
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    symbols[node.target.id] = {"kind": "assign"}
                continue
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
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                symbols[node.target.id] = {"kind": "assign"}
        entry = {"symbols": symbols}
        if detailed:
            entry["regexes"] = _pattern_literals(tree)
            entry["compiled"] = _compiled_patterns(mod_name)
        inventory[mod_name] = entry
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

# Fragments spliced by the fuzzer: every serialization the parsers claim to handle, plus
# the shapes that historically broke them.
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
    "<|message_model|>get_weather<|content_invoke_tool_json|>"
    '{"name": "get_weather", "args": {"city": "Paris"}}<|end_message|>',
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
    '{"city": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
    "<｜tool▁calls▁begin｜>function<｜tool▁sep｜>get_weather\n"
    '```json\n{"city": "Paris"}\n```<｜tool▁calls▁end｜>',
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
    '<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|>'
    "<|tool_calls_section_end|>",
    "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>",
    # One format's markup inside another's argument value: without these, a swap of two
    # arms inside ``strip_segment`` is invisible.
    "<function=x><tool_call> txt <arg_key>c</arg_key></function><parameter=p>",
    '<tool_call>{"name": "search", "arguments": {"q": "<function=get_weather>"}}</tool_call>',
    '<tool_call>{"name": "search", "arguments": {"q": "[TOOL_CALLS]other[ARGS]{}"}}</tool_call>',
    "<function=get_weather><parameter=city><|tool_call>call:search{q:1}<tool_call|></parameter></function>",
    '[TOOL_CALLS]search[ARGS]{"q": "<tool_call>{}</tool_call>"}',
    '<|tool_call>call:search{q:<|"|><think>x</think><|"|>}<tool_call|>',
    '<function=x><parameter=p>get_weather[ARGS]{"a": 1}</parameter></function> tail',
    "<tool_call>get_weather\n<arg_key>c</arg_key>\n<arg_value><function=y></arg_value>\n</tool_call>",
    "<think>I should call get_weather[ARGS]{} but not really</think>",
    "<think>unclosed reasoning",
    '```\nget_weather[ARGS]{"city": "Paris"}\n```',
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


# One fixture per required positional parameter beyond the text. Offsets are derived from
# the text so the balanced scanners start on a real delimiter; without these the driver
# returns ``<undrivable>`` and the digest pins nothing.
_ARG_FIXTURES = {
    "brace_start": lambda text: max(text.find("{"), 0),
    "brace_pos": lambda text: max(text.find("{"), 0),
    "start": lambda text: max(text.find("["), 0),
    "pos": lambda text: len(text),
    "body_start": lambda text: max(text.find("[") + 1, 0),
    "body_end": lambda text: len(text),
    "body": lambda text: text,
    "hard_stop": lambda text: len(text),
    "i": lambda text: 0,
    "idx": lambda text: 0,
    # Scan origin of a strip pass, 0 as the Gemma strip does.
    "floor": lambda text: 0,
    "p": lambda text: 0,
    "n": lambda text: len(text),
    "vs": lambda text: 0,
    "needle": lambda text: "[",
    # ``_safe_cut`` wants the real first-sentinel offset: at 0 it returns 0 for every
    # input and pins nothing.
    "first": lambda text: _parser_first_sentinel(text),
    "found": lambda text: max(_parser_first_sentinel(text), 0),
    "out": lambda text: [],
    # The model-facing notices added for a small window: each takes the tool name first
    # (which gets the corpus text) and then the result it is appended to.
    "result": lambda text: text,
    "last_result": lambda text: text,
    # A run length, not text. Above _MAX_IDENTICAL_TOOL_RESULTS so the message renders
    # the plural branch it will really be seen in.
    "times": lambda text: 3,
    "previous": lambda text: text,
    "markers": lambda text: _tool_healing_build_markers(text),
    "patterns": lambda text: _tool_healing_all_pats(),
    "strip_segment": lambda text: (lambda segment, is_last: segment),
}


# Offsets a predicate is asked about. One offset is not coverage for a function whose job
# is to answer differently at different positions.
_SWEEP_PARAMS = frozenset({"pos"})

# Marks a driver result that holds one entry per boolean variant.
_VARIANTS_KEY = "@variants"

# Boolean parameters driven at both values rather than at one.
_BOTH_WAYS = ("final", "seg_final", "with_spans", "allow_incomplete", "gemma_quotes")

# Function name -> what to hand it in place of the raw corpus entry.
_TEXT_ADAPTERS = {"_gemma_arguments_to_json": lambda text: _gemma_argument_body(text)}


def _sweep_offsets(text: str):
    n = len(text)
    return sorted({0, n // 4, n // 2, (3 * n) // 4, n})


def _gemma_argument_body(text: str) -> str:
    """A raw Gemma argument body out of ``text``, which is what this parser takes.

    Handed a whole corpus entry it almost always raises ``JSONDecodeError``, so its digest
    pinned the exception rather than the key quoting and array normalization it exists for.
    """
    from core import tool_healing

    brace = text.find("{")
    if brace >= 0:
        end = tool_healing._balanced_brace_end(text, brace)
        if end is not None:
            return text[brace + 1 : end]
    return "city:Paris, n:2, tags:[a, b]"


def _parser_first_sentinel(text: str):
    from core.inference import tool_call_parser
    return tool_call_parser._first_sentinel(text, 0)


def _tool_healing_build_markers(text: str):
    from core import tool_healing
    return tool_healing._build_markers(text)


def _tool_healing_all_pats():
    from core import tool_healing
    return tool_healing._TOOL_ALL_PATS


def _drive(func, text: str):
    """Call ``func`` with ``text`` however its signature wants it.

    Returns a JSON-safe result, or a marker string when the function raises: a refactor
    that stops raising, or starts, is a change.
    """
    params = inspect.signature(func).parameters
    names = list(params)
    kwargs = {}
    if "enabled_tool_names" in params:
        kwargs["enabled_tool_names"] = set(_ENABLED_NAMES)
    # Keyword-only, so the positional loop below skips them. Driven at BOTH values:
    # pinning ``final = True`` says nothing about the streaming path.
    combos = [{}]
    for flag in _BOTH_WAYS:
        if flag in params:
            combos = [dict(combo, **{flag: value}) for combo in combos for value in (True, False)]
    if "id_offset" in params:
        kwargs["id_offset"] = 0

    # A few take something derived from the text; a whole corpus entry only pins a raise.
    adapter = _TEXT_ADAPTERS.get(getattr(func, "__name__", ""))
    args = [adapter(text) if adapter else text]
    sweep = None
    # Index arguments come from the first plausible offset, not 0, so the balanced
    # scanners are exercised on a real opening delimiter.
    for extra in names[1:]:
        if extra in kwargs or params[extra].kind == inspect.Parameter.KEYWORD_ONLY:
            continue
        if params[extra].default is not inspect.Parameter.empty:
            continue
        fixture = _ARG_FIXTURES.get(extra)
        if fixture is None:
            return "<undrivable>"
        if extra in _SWEEP_PARAMS:
            sweep = extra
            args.append(None)
            continue
        args.append(fixture(text))

    def _call(extra):
        merged = dict(kwargs, **extra)
        if sweep is not None:
            slot = names.index(sweep)
            results = []
            for offset in _sweep_offsets(text):
                args[slot] = offset
                try:
                    results.append(_jsonable(func(*args, **merged)))
                except Exception as exc:  # noqa: BLE001
                    results.append(f"<raised {type(exc).__name__}>")
            return results
        try:
            outcome = func(*args, **merged)
        except Exception as exc:  # noqa: BLE001 - the exception type is the pinned value
            return f"<raised {type(exc).__name__}>"
        if "out" in names and args[names.index("out")]:
            return [_jsonable(outcome), _jsonable(args[names.index("out")])]
        return _jsonable(outcome)

    if len(combos) > 1:
        # Tagged: plenty of guarded functions return a dict of their own.
        return {_VARIANTS_KEY: {json.dumps(c, sort_keys = True): _call(c) for c in combos}}
    return _call(combos[0])


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
        # A generator's repr carries its address; the yielded values are the behaviour.
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

    Storing every output would be a 7 MB file for no extra guarantee. The corpus is
    rebuilt from ``build_corpus`` rather than checked in, so its own digest is recorded
    too and a corpus edit reports as that instead of as 40 behaviour changes.
    """
    out = {"corpus": _digest([len(corpus), corpus])}
    for mod_name in BEHAVIOUR_MODULES:
        per_module = {}
        for name, func in _guarded_functions(mod_name):
            per_module[name] = _digest([_drive(func, text) for text in corpus])
        out[mod_name] = per_module
    return out


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys = True, ensure_ascii = False, default = repr)
    return hashlib.sha256(payload.encode()).hexdigest()


def first_divergence(corpus, mod_name, func_name):
    """Recompute one function's outputs so a digest mismatch can be localised.

    Run it on both sides of the revision that moved the digest and compare.
    """
    module = importlib.import_module(mod_name)
    func = getattr(module, func_name)
    return [{"input": text, "output": _drive(func, text)} for text in corpus]


def _variants(result):
    """``(label, value)`` per boolean variant the driver produced, or one unlabelled pair.

    Keyed off the tag, not off "is it a dict": a parser returning ``{}`` is a result, not
    a set of variants, and reading it as one reports a meaningless failure.
    """
    if isinstance(result, dict) and set(result) == {_VARIANTS_KEY}:
        return sorted(result[_VARIANTS_KEY].items())
    return [("", result)]


def idempotence_failures(corpus) -> list:
    """``f(f(x)) != f(x)`` for the str -> str functions.

    A stripper that is not idempotent produces different display text depending on how a
    stream is chunked.
    """
    failures = []
    for mod_name in BEHAVIOUR_MODULES:
        for name, func in _guarded_functions(mod_name):
            # Only the strip family is a text -> text projection; feeding a parser's
            # output back into it proves nothing.
            if "strip" not in name or "parse" in name:
                continue
            witnessed = set()
            labels = None
            for text in corpus:
                # One entry per variant; skipping non-strings here would skip the
                # centralized strippers this check is for.
                results = _variants(_drive(func, text))
                if labels is None:
                    labels = {variant for variant, _ in results}
                for variant, once in results:
                    # One witness per (function, variant): a ``final = True`` failure must
                    # not stand in for the streaming path.
                    if variant in witnessed:
                        continue
                    if not isinstance(once, str) or once.startswith("<raised "):
                        continue
                    twice = dict(_variants(_drive(func, once))).get(variant)
                    if twice != once:
                        failures.append(
                            {
                                "module": mod_name,
                                "function": name,
                                "variant": variant,
                                "input": text,
                                "once": once,
                                "twice": twice,
                            }
                        )
                        witnessed.add(variant)
                if labels is not None and witnessed >= labels:
                    break  # every variant already has a witness
    return failures


# ─────────────────────────── 3. Patch-target routing ───────────────────────────

# Every first-party top-level package: stopping at core/routes/utils/state silently
# skipped 32 live targets under hub, storage and picker.
_PATCH_TARGET_RE = re.compile(
    r"""(?:mock\.)?(?:patch|monkeypatch\.setattr)\(\s*"""
    r"""["']((?:core|routes|utils|state|hub|storage|picker)\.[\w.]+)["']"""
)


def patch_targets(tests_dir = None) -> dict:
    """String patch targets found in the test suite, grouped by module."""
    tests_dir = tests_dir or (BACKEND_ROOT / "tests")
    targets = {}
    for path in sorted(tests_dir.rglob("test_*.py")):
        for match in _PATCH_TARGET_RE.finditer(path.read_text(encoding = "utf-8", errors = "ignore")):
            dotted = match.group(1)
            # ``as_posix``: the native form gives backslashes on Windows, so an identical
            # checkout would read as a changed inventory.
            targets.setdefault(dotted, []).append(path.relative_to(BACKEND_ROOT).as_posix())
    return targets


def unresolvable_patch_targets(targets = None) -> list:
    """Targets that no longer resolve to an attribute of an importable module.

    Splits ``a.b.c`` at every dot: the longest importable prefix is the module, the
    remainder must be reachable by ``getattr``.
    """
    targets = targets if targets is not None else patch_targets()
    broken = []
    for dotted, users in sorted(targets.items()):
        parts = dotted.split(".")
        obj = None
        last_error = None
        for split in range(len(parts) - 1, 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:split]))
            except Exception as exc:  # noqa: BLE001 - an unimportable prefix is not the module
                last_error = exc
                continue
            rest = parts[split:]
            break
        else:
            # Nothing imported, and *why* decides what this is: a ModuleNotFoundError
            # naming a prefix of the target means the module is gone, naming anything
            # else means a dependency is absent here. A package whose ``__init__`` pulls
            # in an optional dependency makes every prefix raise.
            environment = not (
                isinstance(last_error, ModuleNotFoundError)
                and last_error.name in {".".join(parts[:i]) for i in range(1, len(parts))}
            )
            entry = {
                "target": dotted,
                "reason": (
                    f"no importable module prefix, unimportable here: {last_error!r}"
                    if environment
                    else "no importable module prefix"
                ),
                "tests": users,
            }
            if environment:
                entry["environment"] = True
            broken.append(entry)
            continue
        for attr in rest:
            # ``core.inference`` resolves attributes through a PEP 562 ``__getattr__``, so
            # a missing optional dependency surfaces as ImportError, not AttributeError.
            try:
                found = hasattr(obj, attr)
            except Exception as exc:  # noqa: BLE001
                # AttributeError = not exported, a dead target. Only ImportError is an
                # environment gap.
                environment = not isinstance(exc, AttributeError)
                entry = {
                    "target": dotted,
                    "reason": (
                        f"unresolvable in this environment: {type(exc).__name__}"
                        if environment
                        else f"lazy export {attr!r} is gone: {exc}"
                    ),
                    "tests": users,
                }
                if environment:
                    entry["environment"] = True
                broken.append(entry)
                break
            if not found:
                # A missing name on a package is ambiguous: gone, or defined in a
                # submodule that is unimportable here. Import it directly to tell apart.
                reason = f"missing attribute {attr!r}"
                environment = False
                if inspect.ismodule(obj) and hasattr(obj, "__path__"):
                    candidate = f"{obj.__name__}.{attr}"
                    try:
                        importlib.import_module(candidate)
                    except ModuleNotFoundError as exc:
                        # Only a failure to import something *else* is environmental.
                        if exc.name and exc.name != candidate:
                            reason = f"submodule {attr!r} needs {exc.name!r}, absent here"
                            environment = True
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

# Names defined in both modules. Unifying them is the point of the refactor; this reports
# where they disagree on real input.
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
            # -1 and None are both "no match"; equate them so only real disagreement shows.
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
    BASELINE_DIR.mkdir(parents = True, exist_ok = True)
    path = BASELINE_DIR / name
    path.write_text(
        json.dumps(payload, indent = 2, sort_keys = True, ensure_ascii = False) + "\n",
        encoding = "utf-8",
    )
    return path


def _read(name):
    return json.loads((BASELINE_DIR / name).read_text(encoding = "utf-8"))


def _diff(
    label,
    old,
    new,
    *,
    additions_matter = True,
):
    """Report the first differing JSON path, which is enough to locate the change.

    ``additions_matter = False`` reports only what changed or disappeared. The question
    these surfaces are asked is "did this refactor drop or alter something", and a later
    unrelated commit adding a symbol or a patch is not a regression; a guard that fails
    on those trains everyone to re-snapshot without reading.
    """
    if old == new:
        return []
    problems = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old:
                if additions_matter:
                    problems.append(f"{label}.{key}: added")
            elif key not in new:
                problems.append(f"{label}.{key}: REMOVED")
            else:
                problems.extend(
                    _diff(f"{label}.{key}", old[key], new[key], additions_matter = additions_matter)
                )
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for index, (a, b) in enumerate(zip(old, new)):
            problems.extend(_diff(f"{label}[{index}]", a, b, additions_matter = additions_matter))
    elif (
        isinstance(old, list)
        and isinstance(new, list)
        and all(isinstance(v, str) for v in old + new)
    ):
        # A name or occurrence list: report what joined or left, not both full lists.
        # Counted, not set-compared, so dropping one of several identical entries (a
        # partial repoint) is not read as "unchanged".
        old_counts, new_counts = Counter(old), Counter(new)
        for name in sorted(set(old) | set(new)):
            delta = new_counts[name] - old_counts[name]
            if delta > 0:
                if additions_matter:
                    problems.append(f"{label}: added {name!r}")
            elif delta < 0:
                if new_counts[name]:
                    problems.append(f"{label}: {old_counts[name]} -> {new_counts[name]} x {name!r}")
                else:
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


def _ast_problems() -> list:
    """AST diff, strict for the modules this branch owns and additive-tolerant elsewhere."""
    recorded = _read("ast_inventory.json")
    live = ast_inventory()
    problems = []
    for mod_name in sorted(set(recorded) | set(live)):
        problems += _diff(
            f"ast.{mod_name}",
            recorded.get(mod_name, {}),
            live.get(mod_name, {}),
            additions_matter = mod_name in BEHAVIOUR_MODULES,
        )
    return problems


def verify() -> int:
    corpus = build_corpus()
    problems = []
    problems += _ast_problems()
    problems += _diff("runtime", _read("runtime_inventory.json"), runtime_inventory())
    problems += _diff("golden", _read("golden_outputs.json"), golden_outputs(corpus))

    # Keyed by variant: a ``final = True`` failure is no licence for a new one on the
    # ``final = False`` streaming path.
    baseline_idempotence = {
        (f["module"], f["function"], f.get("variant", ""))
        for f in _read("idempotence_baseline.json")
    }
    for failure in idempotence_failures(corpus):
        key = (failure["module"], failure["function"], failure["variant"])
        if key not in baseline_idempotence:
            problems.append(
                f"idempotence.{failure['module']}.{failure['function']}"
                f"{'[' + failure['variant'] + ']' if failure['variant'] else ''}: "
                f"f(f(x)) != f(x) for {failure['input']!r}"
            )

    # The recorded set matters too: a patch repointed at another resolvable namespace, or
    # dropped, resolves fine and would otherwise pass.
    recorded = {target: sorted(tests) for target, tests in _read("patch_targets.json").items()}
    live = {target: sorted(tests) for target, tests in patch_targets().items()}
    problems += _diff("patch-targets", recorded, live, additions_matter = False)

    for broken in unresolvable_patch_targets(live):
        # An uninstalled optional backend is not a broken target: ``verify`` has to stay
        # usable in a slim environment.
        if not broken.get("environment"):
            problems.append(
                f"patch-target {broken['target']}: {broken['reason']} ({broken['tests'][0]})"
            )

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
    print(json.dumps(report, indent = 2, ensure_ascii = False))


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
